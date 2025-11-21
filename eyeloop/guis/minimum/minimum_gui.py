"""Minimum GUI for Eyeloop module."""

import os
import threading

import numpy as np
import cv2

import eyeloop.config as config
from eyeloop.constants.minimum_gui_constants import *
from eyeloop.utilities.general_operations import to_int, tuple_int
from vr_core.utilities.logger_setup import setup_logger

class GUI:
    """Minimum GUI for Eyeloop module."""
    def __init__(self) -> None:
        self.display_gui = True

        self.logger = setup_logger("Eyeloop GUI")

        self.side = config.arguments.side

        self.preview = self.side + "_preview"
        self.binary = self.side + "_binary"

        self.dx = 0
        self.dy = 0
        self.cycle = 1
        self.circle_size = 1
        self.locked = False

        self._state = "adjustment"
        self.inquiry = "none"
        self.terminate = -1
        self.update = self.adj_update#real_update
        self.skip = 0
        self.first_run = True


    def release(self):
        cv2.destroyAllWindows()


    def arm(self, width: int, height: int) -> None:
        self.fps = np.round(1/config.arguments.fps, 2)

        self.pupil_processor = config.engine.pupil_processor

        if not self.display_gui:
            return

        self.width, self.height = width, height
        self.binary_width = max(width, 300)
        self.binary_height = max(height, 200)

        self.bin_stock = np.zeros((self.binary_height, self.binary_width))
        self.bin_P = self.bin_stock.copy()

        self.src_txt = np.zeros((20, width, 3))
        self.prev_txt = self.src_txt.copy()

        self.bin_stock_txt = np.zeros((20, self.binary_width))
        self.bin_stock_txt_selected = self.bin_stock_txt.copy()

        if (self.side == "Right"):
            x_shift = 350
        else:
            x_shift = 25

        cv2.namedWindow(self.preview, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.binary, cv2.WINDOW_NORMAL)

        cv2.imshow(self.preview, np.hstack((self.bin_stock, self.bin_stock)))
        cv2.imshow(self.binary, np.vstack((self.bin_stock, self.bin_stock)))

        cv2.resizeWindow(self.preview, 300, 450)
        cv2.resizeWindow(self.binary, 300, 450)

        cv2.moveWindow(self.preview, x_shift, 25)
        cv2.moveWindow(self.binary, x_shift, 500)


    def place_cross(self, source: np.ndarray, point: tuple, color: tuple) -> None:
        try:
            source[to_int(point[1] - 20):to_int(point[1] + 19), to_int(point[0]-1):to_int(point[0]+1)] = color
            source[to_int(point[1]-1):to_int(point[1]+1), to_int(point[0] - 20):to_int(point[0] + 19)] = color
        except:
            pass


    def skip_track(self):
        self.update = self.real_update


    def pupil(self, source_rgb):
        try:
            pupil_center, pupil_width, pupil_height, pupil_angle = config.engine.dataout["pupil"]
            #self.logger.info("pupil radius: %s", pupil_width)
            cv2.ellipse(source_rgb, tuple_int(pupil_center), tuple_int((pupil_width, pupil_height)), pupil_angle, 0, 360, red, 1)
            self.place_cross(source_rgb, pupil_center, red)
            return True
        except Exception as e:
            #print(f"pupil not found: {e}")
            return False


    def adj_update(self, img):
        source_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self.bin_P = self.bin_stock.copy()

        if self.pupil(source_rgb):
            self.bin_P[0:20, 0:self.binary_width] = self.bin_stock_txt_selected
        else:
            self.bin_P[0:20, 0:self.binary_width] = self.bin_stock_txt

        try:
            pupil_area = self.pupil_processor.source

            offset_y = int((self.binary_height - pupil_area.shape[0]) / 2)
            offset_x = int((self.binary_width - pupil_area.shape[1]) / 2)
            self.bin_P[offset_y:min(offset_y + pupil_area.shape[0], self.binary_height),
            offset_x:min(offset_x + pupil_area.shape[1], self.binary_width)] = pupil_area
        except:
            pass

        if self.display_gui:
            cv2.imshow(self.preview, source_rgb)
            cv2.imshow(self.binary, self.bin_P)
            cv2.waitKey(50)

        if self.first_run:
                #cv2.destroyAllWindows()
                self.first_run = False
                self.centre = (round(self.binary_width/2), round(self.binary_height/2))
                self.cursor = self.centre
        # else:
        #     #self.logger.info("locked: %s; auto_search: %s", self.locked, config.arguments.auto_search)
        #     if self.locked == False and config.arguments.auto_search == True:
        #         self.pupil_lock()


    def real_update(self, img) -> None:
        source_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.pupil(source_rgb)

        if self.display_gui:
            cv2.imshow(self.preview, source_rgb)
            cv2.waitKey(1)
        threading.Timer(self.fps, self.skip_track).start() #run feed every n secs (n=1)
        self.update = lambda _: None


    def pupil_lock(self):
        """
        This method tries to lock on to the pupil. If sucessful it initiates the tracking algorithm.
        If not, it calls center_offset_generater() to adjust the cursor value.
        """

        self.logger.info("<%s> attempting to lock pupil", config.arguments.side)

        try:
            # If sucessful, tracking is initiated
            if (self.pupil_processor.fit_model.params[1] > config.arguments.min_radius_threshold and
                self.pupil_processor.fit_model.params[1] < config.arguments.max_radius_threshold):

                self.logger.info("<%s> pupil is locked.", config.arguments.side)

                self.locked = True
                self.inquiry = "track"

                self._state = "tracking"

                self.update = self.real_update

                return
            else:
                # If not sucessful, cursor adjustment is made
                self.logger.info("<%s> pupil is not locked, attempting to lock.", config.arguments.side)
                self.center_offset_generator()

        except:
            pass

        # Tries to lock on to the pupil with the current cursor value
        self.pupil_processor.reset(self.cursor)


    def center_offset_generator(self):
        """
        This method changes the value of cursor for searching the pupil.
        It circles (moves in a square) around the centre of the images.
        The position difference between the new and old cursor value is always in size of step.
        After finishing a whole circle (square) a new and bigger one
        will initiate with radius of step * self.circle_size.
        Current position of the square is given by self.cycle
        """

        # Square value computation---------------------------------------------

        step = config.arguments.search_step # Step size for the search

        if self.cycle == 1:                             # Initial position
            self.dx = - step * self.circle_size
            self.dy = - step * self.circle_size
            #print("x: " + str(self.dx) + ", y: " + str(self.dy))
        elif self.cycle < (4 + 2*(self.circle_size-1)): # Top side
            self.dx += step
        elif self.cycle < (6 + 4*(self.circle_size-1)): # Right side
            self.dy += step
        elif self.cycle < (8 + 6*(self.circle_size-1)): # Bottom side
            self.dx -= step
        else:                                           # Left side
            self.dy -= step


        # Value assignment-----------------------------------------------------

        # Adding a new value to cursor
        self.cursor = (self.centre[0] + self.dx, self.centre[1] + self.dy)

        self.logger.info("<%s> cursor new value: %s", config.arguments.side, self.cursor)

        # Return if cursor is beyond window
        if self.cursor[0] > self.binary_width | self.cursor[1] > self.binary_height:
            print("No pupil find, exiting.")
            return

        # Iteration------------------------------------------------------------

        # Check whether a whole circle around centre has been made
        if self.cycle % (8 * self.circle_size) == 0:
            self.circle_size += 1 # Next circle will be bigger
            self.cycle = 1 # Reset cycle count
            return

        # Increase cycle count for next search
        self.cycle += 1

        #self.pupil_processor.reset(self.cursor)
