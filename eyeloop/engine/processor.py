# ruff: noqa: F403, F405

"""Processor module for eye features (pupil, corneal reflection)."""

import collections

import numpy as np
import cv2

import eyeloop.config as config
from eyeloop.constants.processor_constants import *
from eyeloop.engine.models.circular import Circle
from eyeloop.engine.models.ellipsoid import Ellipse
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

        radius_buffer_size = 30
        self.radius_drop_factor = 0.85

        self.radius_buffer = collections.deque(maxlen=radius_buffer_size)
        self.filtered_radius = None
        self.filtered_center = None

        self.walkout_offset = 0

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
            self.last_min_radius = self.min_radius

        self.logger.info("Min_radius = %s", self.min_radius)
        self.compute_threshold()


    def compute_threshold(self) -> None:
        """Computes the threshold for pupil detection based on min_radius."""
        self.threshold = len(crop_stock) * self.min_radius * 1.05


    def track(self, source):
        if self.last_min_radius != self.min_radius:
            self.compute_threshold()
            self.last_min_radius = self.min_radius

        self.raw = source
        self.source = source.copy()

        # Performs a simple binarization and applies a smoothing gaussian kernel.
        self.pupil_thresh() #either pupil or cr

        mean_img = np.mean(self.source)

        try:
            config.blink[config.blink_i] = mean_img
            config.blink_i += 1
            self.blink_sampled(1)

        except IndexError:
            self.blink_sampled(0)
            self.blink_sampled = lambda _: None
            config.blink_i = 0

        baseline = np.mean(config.blink[np.nonzero(config.blink)])
        self.dataout = {}
        diff = np.abs(mean_img - baseline)

        # self.logger.info("Mean image intensity: %.2f, baseline: %.2f, diff: %.2f", mean_img, baseline, diff)

        if diff > 1:
            config.engine.dataout[self.type_entry] = None
            # self.logger.info("Blink detected.")
            return

        self.fit() #gets fit model


    def blink_sampled(self, t: int = 1):
        """Calibrates blink detection based on sampled mean image intensity."""

        # if t == 1:
        #     if config.blink_i % 20 == 0:
        #         print(f"calibrating blink detector "
        #             f"{round(config.blink_i/config.blink.shape[0]*100,1)}%")
        # else:
        #     self.logger.info("(success) blink detection calibrated")


    def pupil_thresh(self):
        self.source[:] = cv2.threshold(
            cv2.GaussianBlur(
                cv2.erode(
                    self.source, kernel, iterations = 1
                ), self.blur, 0), self.binarythreshold, 255, cv2.THRESH_BINARY_INV
        )[1]


    def fit(self):
        try:
            r = self.pupil_walkout()

            self.center = self.fit_model.fit(r)
            # raw_r = (self.fit_model.params[1] + self.fit_model.params[2]) / 2.0
            frame_valid = self.radius_filter()

            if frame_valid:
                # normal tracking output
                config.engine.dataout[self.type_entry] = self.fit_model.params
            else:
                # snap: return empty output
                config.engine.dataout[self.type_entry] = None

            # if config.arguments.side == "Right":
            #     self.logger.info("raw=%.3f filtered=%.3f", raw_r, self.fit_model.params[1])

        except IndexError:
            # self.logger.info("Fit index error")
            self.center_adj()

        except Exception as e:
            # self.logger.info(f"Fit-func error: {e}")
            self.center_adj()


    def radius_filter(self) -> bool:
        """Filters the radius to avoid sudden jumps."""
        cxcy, rx, ry, ang = self.fit_model.params
        new_radius = (rx + ry) / 2.0
        filtered_center = cxcy

        self.radius_buffer.append(new_radius)

        is_snap = False

        # If we have at least 3 previous samples, compare against their mean
        if len(self.radius_buffer) > 3:
            # Mean of previous radii (exclude the newest one)
            prev_radii = list(self.radius_buffer)[:-1]
            mean_radius = float(np.mean(prev_radii))

            if new_radius < mean_radius * self.radius_drop_factor:
                is_snap = True

                # Suspiciously small radius: KEEP last filtered radius in the output
                if self.filtered_radius is not None:
                    filtered_r = self.filtered_radius
                else:
                    # First time we see a snap: fall back to mean of history
                    filtered_r = mean_radius

                # And also keep last valid center if we have one
                if self.filtered_center is not None:
                    filtered_center = self.filtered_center
            else:
                # Looks ok -> accept
                filtered_r = new_radius

        # Remember last accepted radius
        self.filtered_radius = filtered_r
        self.filtered_center = filtered_center

        self.center = filtered_center
        self.fit_model.params = (filtered_center, filtered_r, filtered_r, ang)

        return not is_snap


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

        # self.logger.info("1. Crop_list sum: %s; threshold: %s", np.sum(crop_list), self.threshold)

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

            # self.logger.info("2. Crop_list sum: %s; threshold: %s", np.sum(crop_list), self.threshold)

            if np.sum(crop_list) < self.threshold:
                # self.logger.warning("Pupil walkout failed: insufficient edge points found.")
                raise IndexError

        r[:8,:] = center
        r[ry_add, 1] += crop_list[ry_add]
        r[rx_add, 0] += crop_list[rx_add]
        r[ry_subtract, 1] -= crop_list[ry_subtract] #
        r[rx_subtract, 0] -= crop_list[rx_subtract]
        r[rx_multiplied, 0] *= rx_multiply
        r[ry_multiplied, 1] *= ry_multiply
        r[8:,:] += center

        return self.cond(r, crop_list)


    def cond(self, r, crop_list):
        dists =  np.linalg.norm(np.mean(r,  axis = 0,dtype=np.float64) - r, axis = 1)

        mean_ = np.mean(dists)
        std_ = np.std(dists)

        lower, upper = mean_ - std_, mean_ + std_ * .8
        cond_ = np.logical_and(np.greater_equal(dists, lower), np.less(dists, upper))

        return r[cond_]


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
                self.center = (self.raw.shape[1]//2, self.raw.shape[0]//2)

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
