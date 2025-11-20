"""Processor module for eye features (pupil, corneal reflection)."""

import numpy as np
import cv2

import eyeloop.config as config
from eyeloop.constants.processor_constants import *
from eyeloop.engine.models.circular import Circle
from eyeloop.engine.models.ellipsoid import Ellipse
from eyeloop.utilities.general_operations import to_int, tuple_int
from vr_core.utilities.logger_setup import setup_logger


class Center_class():
    """Center processor for eye features (pupil, corneal reflection)."""

    def fit(self, r):

        self.params = tuple(np.mean(r, axis = 0))
        return self.params


class Shape():
    """Shape processor for eye features (pupil, corneal reflection)."""
    def __init__(self, type = 1, n = 0):
        self.side = config.arguments.side

        self.logger = setup_logger(f"{self.side} processor")
        self.active = False
        self.center = -1

        self.walkout_offset = 0

        self.last_walkout_points = 0

        self.binarythreshold = -1
        self.blur = (3, 3)
        self.type = type

        self.model = config.arguments.model
        self.type_entry = None

        self.side = config.arguments.side

        if type == 1:
            self.type_entry = "pupil"

            if self.model == "circular":
                self.fit_model = Circle(self)
            else:
                self.fit_model = Ellipse(self)

            self.min_radius = 2
            self.max_radius = 100 #change according to video size or argument

        self.threshold = len(crop_stock) * self.min_radius *1.05


    def pupil_thresh(self):
        self.source[:] = cv2.threshold(
            cv2.GaussianBlur(
                cv2.erode(
                    self.source, kernel, iterations = 1
                ), self.blur, 0), self.binarythreshold, 255, cv2.THRESH_BINARY_INV
        )[1]


    def reset(self, center):
        self.logger.info("Resetting processor with center: %s", center)
        self.active = True
        self.margin = 0
        self.walkout_offset = 0
        self.center = center

        self.standard_corners = [(0, 0), (config.engine.width, config.engine.height)]

        self.corners = self.standard_corners.copy()


    def track(self, source):
        self.raw = source
        self.source = source.copy()

        # Performs a simple binarization and applies a smoothing gaussian kernel.
        self.pupil_thresh() #either pupil or cr
        self.fit() #gets fit model


    def center_adj(self):
        #adjust settings:
        # blurred = cv2.GaussianBlur(self.raw, (3, 3), 2)
        circles = cv2.HoughCircles(self.raw, cv2.HOUGH_GRADIENT, 1.5, 10, param1=200, param2=15, minRadius=self.min_radius, maxRadius=self.max_radius)

        if circles is None:
            # self.logger.info("No circles found for center adjustment.")
            return
        else:
            smallest = -1
            current = -1

            if self.center == -1:
                center = (self.raw.shape[1]//2, self.raw.shape[0]//2)
                self.reset(center)

            for circle in circles[0, :]:
                score = (
                    self.distance(circle[:2], self.center) +
                    np.mean(
                        self.raw[int(circle[1])-self.min_radius:int(circle[1])+self.min_radius,
                                 int(circle[0]-self.min_radius):int(circle[0]+self.min_radius)]
                        ))

                self.raw[int(circle[1]), int(circle[0])] = 100
                # cv2.imshow("kk", self.raw)
                # cv2.waitKey(0)
                if smallest == -1:
                    smallest = score
                    current = circle[:2]
                elif score < smallest:
                    smallest = score
                    current = circle[:2]

            self.center = tuple(current)
            # self.logger.info("Image shape: %s; Found center: %s", self.raw.shape, self.center)


    def distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


    def artefact_(self, params):
        cv2.circle(config.engine.pup_source, tuple_int(params[0]), to_int(params[1] * self.expand), black, -1)


    def fit(self):
        try:
            r = self.pupil_walkout()

            self.center = self.fit_model.fit(r)
            #self.logger.info(params[1])
            #self.logger.info(self.last_walkout_points)

            config.engine.dataout[self.type_entry] = self.fit_model.params

        except IndexError:
            self.logger.info("Fit index error")
            self.center_adj()

        except Exception as e:
            self.logger.info(f"Fit-func error: {e}")
            self.center_adj()


    def cond(self, r, crop_list):
        dists =  np.linalg.norm(np.mean(r,  axis = 0,dtype=np.float64) - r, axis = 1)

        mean_ = np.mean(dists)
        std_ = np.std(dists)

        lower, upper = mean_ - std_, mean_ + std_ * .8
        cond_ = np.logical_and(np.greater_equal(dists, lower), np.less(dists, upper))

        return r[cond_]


    def clip(self, crop_list):
        np.clip(crop_list, self.min_radius, self.max_radius, out = crop_list)


    def pupil_walkout(self):
        try:
            center = np.round(self.center).astype(int)
            # self.logger.info("Pupil walkout with center: %s", center)
        except:
            return

        canvas = np.array(self.source, dtype=int)
        canvas[-1,:] = canvas[:,-1] = canvas[0,:] = canvas[:,0] = 0

        r = rr_2d.copy()

        crop_list = crop_stock.copy()

        canvas_ = canvas[center[1]:, center[0]:]
        canv_shape0, canv_shape1 = canvas_.shape
        crop_canvas = np.flip(canvas[:center[1], :center[0]])
        crop_canv_shape0, crop_canv_shape1 = crop_canvas.shape

        crop_canvas2 = np.fliplr(canvas[center[1]:, :center[0]])
        crop_canv2_shape0, crop_canv2_shape1 = crop_canvas2.shape

        crop_canvas3 = np.flipud(canvas[:center[1], center[0]:])
        crop_canv3_shape0, crop_canv3_shape1 = crop_canvas3.shape

        canvas2 = np.flip(canvas) # flip once

        crop_list=np.array([
            np.argmax(canvas_[:, 0][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas_[0, :][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas2[-center[1], -center[0]:][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas2[-center[1]:, -center[0]][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][self.min_radius:self.max_radius] == 0),
            np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][self.min_radius:self.max_radius] == 0)
        ], dtype=int) + self.min_radius

        #self.logger.info("1. Crop_list sum: %s; threshold: %s", np.sum(crop_list), self.threshold)

        #self.logger.info(crop_list)

        if np.sum(crop_list) < self.threshold:
            #origin inside corneal reflection?
            offset_list = np.array([
                np.argmax(canvas_[:, 0][1:] == 255), np.argmax(canvas_[0, :][1:] == 255),
                np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
                np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
                np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
                np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
                np.argmax(canvas2[-center[1], -center[0]:][1:] == 255), np.argmax(canvas2[-center[1]:, -center[0]][1:] == 255),
                np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
                np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
                np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
                np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
                np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
                np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
                np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
                np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
                np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
                np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
                np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
                np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
                np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
                np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
                np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
                np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
                np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
                np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
                np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
                np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
                np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
                np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
                np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
                np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255)
            ], dtype=int) + 1

            crop_list=np.array([
            np.argmax(canvas_[:, 0][offset_list[0]:] == 0), np.argmax(canvas_[0, :][offset_list[1]:] == 0), np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][offset_list[2]:] == 0),
            np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[3]:] == 0), np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[4]:] == 0),
            np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[5]:] == 0), np.argmax(canvas2[-center[1], -center[0]:][offset_list[6]:] == 0), np.argmax(canvas2[-center[1]:, -center[0]][offset_list[7]:] == 0),
            np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][offset_list[8]:] == 0), np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[9]:] == 0), np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[10]:] == 0),
            np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[11]:] == 0), np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][offset_list[12]:] == 0),
            np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[13]:] == 0), np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[14]:] == 0),
            np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[15]:] == 0), np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][offset_list[16]:] == 0), np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[17]:] == 0),
            np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[18]:] == 0), np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[19]:] == 0), np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][offset_list[20]:] == 0),
            np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[21]:] == 0), np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[22]:] == 0), np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[23]:] == 0),
            np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][offset_list[24]:] == 0), np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[25]:] == 0), np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[26]:] == 0),
            np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[27]:] == 0), np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][offset_list[28]:] == 0), np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[29]:] == 0),
            np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[30]:] == 0), np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[31]:] == 0)
            ], dtype=int) + offset_list

            #self.logger.info("2. Crop_list sum: %s; threshold: %s", np.sum(crop_list), self.threshold)

            if np.sum(crop_list) < self.threshold:
                raise IndexError(f"[WARN] [Processor {self.side}]: Lost track, do reset")

        r[:8,:] = center
        r[ry_add, 1] += crop_list[ry_add]
        r[rx_add, 0] += crop_list[rx_add]
        r[ry_subtract, 1] -= crop_list[ry_subtract] #
        r[rx_subtract, 0] -= crop_list[rx_subtract]
        r[rx_multiplied, 0] *= rx_multiply
        r[ry_multiplied, 1] *= ry_multiply
        r[8:,:] += center

        self.last_walkout_points = len(r)
        return self.cond(r, crop_list)
