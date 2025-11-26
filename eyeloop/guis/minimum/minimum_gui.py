"""Minimum GUI for Eyeloop module."""

import numpy as np
import cv2

import eyeloop.config as config
from eyeloop.constants.minimum_gui_constants import *  # noqa: F403
from eyeloop.utilities.general_operations import to_int, tuple_int
from vr_core.utilities.logger_setup import setup_logger

class GUI:
    """Minimum GUI for Eyeloop module."""
    def __init__(self) -> None:
        self.display_gui = True

        self.logger = setup_logger("Eyeloop GUI")

        self.side = config.arguments.side

        self.preview = self.side + "_preview"
        self.pupil_bin = self.side + "_binary_pupil"
        self.cr_bin = self.side + "_binary_cr"

        self.dx = 0
        self.dy = 0
        self.cycle = 1
        self.circle_size = 1
        self.locked = False
        self.print_cycle = 0

        self._state = "adjustment"
        self.inquiry = "none"
        self.terminate = -1
        self.update = self.adj_update
        self.skip = 0
        self.first_run = True


    def release(self):
        cv2.destroyAllWindows()


    def arm(self, width: int, height: int) -> None:
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

        shape = self.bin_stock.shape
        height, width = shape[0], shape[1]
        scale = 1

        width = width // scale
        height = height // scale

        if (self.side == "Right"):
            x_shift = width
        else:
            x_shift = 0

        cv2.namedWindow(self.preview)
        cv2.namedWindow(self.pupil_bin)

        cv2.imshow(self.preview, np.hstack((self.bin_stock, self.bin_stock)))
        cv2.imshow(self.pupil_bin, np.vstack((self.bin_stock, self.bin_stock)))

        cv2.resizeWindow(self.preview, width, height)
        cv2.resizeWindow(self.pupil_bin, width, height)

        cv2.moveWindow(self.preview, x_shift, 0)
        cv2.moveWindow(self.pupil_bin, x_shift, height + 30)

        if config.engine.cr_processor_1 is not None:
            cv2.namedWindow(self.cr_bin)
            cv2.imshow(self.cr_bin, np.vstack((self.bin_stock, self.bin_stock)))
            cv2.resizeWindow(self.cr_bin, width, height)
            cv2.moveWindow(self.cr_bin, x_shift, 2 * height + 60)


    def place_cross(
        self,
        source: np.ndarray,
        center: tuple[float, float],
        color: tuple[float, float, float],
        thickness: int,
        size: int
    ) -> None:
        try:
            source[to_int(center[1] - size):to_int(center[1] + size-1), to_int(center[0]-thickness):to_int(center[0]+thickness)] = color
            source[to_int(center[1]-thickness):to_int(center[1]+thickness), to_int(center[0] - size):to_int(center[0] + size-1)] = color
        except Exception:
            pass


    def pupil(self, source_rgb):
        if config.engine.dataout["pupil"]:
            try:
                ((pupil_center_x, pupil_center_y), pupil_radius) = config.engine.dataout["pupil"]
                cv2.ellipse(source_rgb, tuple_int((pupil_center_x, pupil_center_y)), tuple_int((pupil_radius, pupil_radius)), 0, 0, 360, red, 1)  # noqa: F405
                self.place_cross(source_rgb, (pupil_center_x, pupil_center_y), red, 1, 20)  # noqa: F405
            except Exception as e:
                self.logger.error(f"Pupil mark error: {e}")
        if config.engine.dataout["cr"]:
            try:
                cr_list = config.engine.dataout["cr"]
                for cr_center in cr_list:
                    self.place_cross(source_rgb, cr_center[0], green, 1, 12)  # noqa: F405
            except Exception as e:
                self.logger.error(f"CR mark error: {e}")


    def adj_update(self, img):

        self.print_cycle += 1
        if self.print_cycle % 100 == 0 and self.display_gui:
            source_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.bin_P = self.bin_stock.copy()
            self.pupil(source_rgb)

            pupil_area = self.pupil_processor.source
            cv2.imshow(self.preview, source_rgb[::2, ::2])
            cv2.imshow(self.pupil_bin, pupil_area[::2, ::2])
            if config.engine.cr_processor_1 is not None:
                cr1_area = config.engine.cr_processor_1.source
                cv2.imshow(self.cr_bin, cr1_area)
            cv2.waitKey(1)
            self.print_cycle = 0

        if self.first_run:
                #cv2.destroyAllWindows()
                self.first_run = False
                self.centre = (round(self.binary_width/2), round(self.binary_height/2))
                self.cursor = self.centre
