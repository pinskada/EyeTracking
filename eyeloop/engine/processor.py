# ruff: noqa: F403, F405

"""Processor module for eye features (pupil, corneal reflection)."""

import collections
import time

import numpy as np
import cv2

import eyeloop.config as config
from eyeloop.constants.processor_constants import *
from eyeloop.engine.models.circular import Circle
from eyeloop.engine.models.ellipsoid import Ellipse  # noqa: F401
from vr_core.utilities.logger_setup import setup_logger


class Center_class():
    """Center processor for eye features (pupil, corneal reflection)."""

    def fit(self, r) -> tuple[float, float]:
        """Fit the center of the given points.

        Args:
            r: (N, 2) ndarray of points
        Returns:
            Center of the given points
        """

        self.params = tuple(np.mean(r, axis = 0))
        return self.params


class Fast_Elliptical:
    """
    Faster, center-focused ellipse fit.

    - Uses a linear least-squares fit of a general conic
        A x^2 + B x y + C y^2 + D x + E y + F = 0
      with F fixed to -1 (so we solve for A..E).
    - Extracts center (cx, cy) from the conic via 2x2 solve.
    - Radius is a robust average distance of points to this center.

    Output:
        params = ((cx, cy), r_est)
    """

    def __init__(self):
        self.params: tuple[tuple[float, float], float] | None = None

    def fit(self, r) -> tuple[tuple[float, float], float]:
        r = np.asarray(r, dtype=np.float64)
        if r.ndim != 2 or r.shape[1] != 2:
            raise ValueError("Fast_Elliptical.fit expects (N, 2) array")
        n_pts = r.shape[0]
        if n_pts < 5:
            # Not enough points for a conic; fall back to simple mean.
            center = tuple(np.mean(r, axis=0))
            radii = np.linalg.norm(r - center, axis=1)
            r_est = float(np.median(radii)) if radii.size > 0 else 0.0
            center = (float(center[0]), float(center[1]))
            self.params = (center, r_est)
            return self.params

        x = r[:, 0]
        y = r[:, 1]

        # Shift to improve numerical stability (we'll shift back later).
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x0 = x - x_mean
        y0 = y - y_mean

        # Design matrix for A,B,C,D,E with F fixed as -1:
        # A x^2 + B x y + C y^2 + D x + E y - 1 = 0  -> RHS = 1
        A_mat = np.column_stack((x0 * x0, x0 * y0, y0 * y0, x0, y0))  # (N, 5)
        b_vec = np.ones_like(x0)  # (N,)

        try:
            # Solve A_mat @ p ≈ b_vec  for p=[A,B,C,D,E]
            p, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            A_c, B_c, C_c, D_c, E_c = p
        except np.linalg.LinAlgError:
            # Fallback: robust mean center + median radius
            center = tuple(np.mean(r, axis=0))
            radii = np.linalg.norm(r - center, axis=1)
            r_est = float(np.median(radii)) if radii.size > 0 else 0.0
            center = (float(center[0]), float(center[1]))
            self.params = (center, r_est)
            return self.params

        # Solve for center in the shifted coordinates (x0, y0):
        # [2A  B ] [cx0] = [-D]
        # [ B 2C] [cy0]   [-E]
        M = np.array([[2.0 * A_c, B_c],
                      [B_c,       2.0 * C_c]], dtype=np.float64)
        rhs = np.array([-D_c, -E_c], dtype=np.float64)

        try:
            center_local = np.linalg.solve(M, rhs)
            cx0, cy0 = center_local
        except np.linalg.LinAlgError:
            # Degenerate case: revert to mean in shifted coords.
            cx0, cy0 = 0.0, 0.0

        # Shift back to original coordinates
        cx = float(cx0 + x_mean)
        cy = float(cy0 + y_mean)
        center = (cx, cy)

        # Robust radius: median-based with outlier rejection
        dists = np.linalg.norm(r - center, axis=1)
        if dists.size == 0:
            r_est = 0.0
        elif dists.size < 5:
            r_est = float(np.mean(dists))
        else:
            med = float(np.median(dists))
            mad = float(np.median(np.abs(dists - med))) + 1e-6  # avoid div-by-zero
            # Keep points within ~2.5 MAD of the median distance
            mask = np.abs(dists - med) < 2.5 * mad
            if np.any(mask):
                r_est = float(np.mean(dists[mask]))
            else:
                r_est = med

        self.params = (center, r_est)
        return self.params


class Shape():
    """Shape processor for eye features (pupil, corneal reflection)."""
    def __init__(self, type = 1) -> None:
        self.side = config.arguments.side

        self.process_blink = False

        self.logger = setup_logger(f"{self.side} processor")
        self.active = False
        self.center = -1

        self.filtered_radius = None
        self.filtered_center = None

        self.walkout_offset = 0

        self.type = type

        #self.model = config.arguments.model
        self.model = "elliptical"
        # self.model = "fast_elliptical"

        self.side = config.arguments.side

        self.dt_fail_count = 0
        self.dt_fail_limit = 3

        if type == 1:
            """Pupil processor settings."""
            self.type_entry = "pupil"
            if self.model == "circular":
                self.fit_model = Circle(self)
            elif self.model == "elliptical":
                self.fit_model = Ellipse(self)
            elif self.model == "fast_elliptical":
                self.fit_model = Fast_Elliptical()
            else:
                self.logger.error(f"Unknown model: {self.model}")

            self.apply_thresh = self.pupil_thresh_

            # Threshold settings ----------------------------------------------
            self.binarythreshold = -1 # Binary threshold (computed later)
            self.blur = (3, 3) # Blur size for thresholding

            # Radius settings -------------------------------------------------
            self.min_radius = 2 # Minimum expected radius for pupil detection
            self.max_radius = 100 # Maximum expected radius for pupil detection

            # Distance transform settings ------------------------------------
            self.circularity_min = 2.0 # Minimum circularity for pupil detection
            self.circularity_max = 3.7 # Maximum circularity for pupil detection
            self.aspect_ratio_min = 0.7 # Minimum aspect ratio for pupil detection

            self.w_r = 0.4   # Score weight for radius
            self.w_c = 0.4   # Score weight for circularity
            self.w_d = 0.2   # Score weight for distance to previous center

            # Radius filter settings -----------------------------------------
            self.radius_drop_factor = 0.0 # Maximum drop factor for radius in one frame
            radius_buffer_size = 20 # Size of the buffer for radius filtering

            self.radius_buffer = collections.deque(maxlen=radius_buffer_size)

        elif type == 2:
            """Corneal reflection processor settings."""
            self.type_entry = "cr"
            self.fit_model = Center_class()
            self.apply_thresh = self.cr_thresh_

            self.number_of_cr = 1 # Set how many CRs to track

            # Threshold settings ----------------------------------------------
            self.binarythreshold = 200 # Binary threshold
            self.blur = (1, 1) # Blur size for thresholding

            # Radius settings -------------------------------------------------
            self.min_radius = 1 # Minimum expected radius for CR detection
            self.max_radius = 5 # Maximum expected radius for CR detection

            # Distance transform settings -------------------------------------
            self.circularity_min = 0.0 # Minimum circularity for CR detection
            self.circularity_max = 100.0 # Maximum circularity for CR detection
            self.aspect_ratio_min = 2 # Minimum aspect ratio for CR detection

            self.w_r = 0.4   # Score weight for radius
            self.w_c = 0.4   # Score weight for circularity
            self.w_d = 0.2   # Score weight for distance to previous center

        else:
            self.logger.error("Unknown processor type: %s", type)

        self.last_min_radius = self.min_radius
        self.compute_threshold()

        self.time_start: float | None = None
        self.time_threshold: float | None = None
        self.time_walkout: float | None = None
        self.time_fit_model: float | None = None
        self.time_radius_filter: float | None = None


    def compute_threshold(self) -> None:
        """Computes the threshold for pupil detection based on min_radius."""
        self.threshold = len(crop_stock) * self.min_radius * 1.05
        self.logger.info(
            "Type: %s_%s; Min_radius: %s; Max_radius: %s; Threshold: %s",
            config.arguments.side,
            self.type_entry,
            self.min_radius,
            self.max_radius,
            self.threshold
        )


    def track(self, source) -> None:
        """Tracks the eye feature in the given source image."""
        self.time_start = time.perf_counter_ns() / 1e9

        if self.last_min_radius != self.min_radius:
            self.compute_threshold()
            self.last_min_radius = self.min_radius

        self.raw = source
        self.source = source.copy()

        # Performs a simple binarization and applies a smoothing gaussian kernel.
        self.apply_thresh() #either pupil or cr

        self.time_threshold = time.perf_counter_ns() / 1e9

        mean_img = np.mean(self.source)

        # if self.type_entry is None:
        #     self.logger.error("Processor type_entry is None.")
        #     return

        if self.process_blink:
            try:
                config.blink[config.blink_i] = mean_img
                config.blink_i += 1

            except IndexError:
                config.blink_i = 0

            baseline = np.mean(config.blink[np.nonzero(config.blink)])
            diff = np.abs(mean_img - baseline)

            # self.logger.info("Mean image intensity: %.2f, baseline: %.2f, diff: %.2f", mean_img, baseline, diff)

            if diff > 5:
                config.engine.dataout[self.type_entry] = ()
                self.logger.info("Blink detected.")
                return
        if self.type_entry == "pupil":
            self.fit()
        else:
            self.center_adj_dt()

        # self._log_timings()


    def pupil_thresh_(self) -> None:
        self.source[:] = cv2.threshold(
            cv2.GaussianBlur(
                cv2.erode(self.source, kernel, iterations = 1),
                self.blur,
                0
            ),
            self.binarythreshold,
            255,
            cv2.THRESH_BINARY_INV,
        )[1]


    def cr_thresh_(self) -> None:
        _, self.source[:] = cv2.threshold(
            cv2.GaussianBlur(self.source, self.blur, 0),
            self.binarythreshold,
            255,
            cv2.THRESH_BINARY
        )


    def fit(self) -> None:
        try:
            r = self.pupil_walkout()
            fit_params = self.fit_model.fit(r)
            # self.logger.info("Pupil fit success.")
            self.time_fit_model = time.perf_counter_ns() / 1e9

            self.center = fit_params[0]

            # raw_r = (self.fit_model.params[1] + self.fit_model.params[2]) / 2.0
            try:
                frame_valid = self.radius_filter()
            # frame_valid = True
            except Exception as e:
                self.logger.error(f"Radius filter error: {e}")
                frame_valid = True

            if self.type_entry is not None and self.fit_model.params is not None:
                if frame_valid:
                    # Normal tracking output
                    config.engine.dataout[self.type_entry] = fit_params
                else:
                    # Snap: return empty output
                    config.engine.dataout[self.type_entry] = ()

            # if config.arguments.side == "Right":
            #     self.logger.info("raw=%.3f filtered=%.3f", raw_r, self.fit_model.params[1])

        except IndexError as e:  # noqa: F841
            # self.logger.info(f"Fit index error: {e}")
            self.backup_fit()

        except Exception as e:  # noqa: F841
            # self.logger.info(f"Fit-func error: {e}")
            self.backup_fit()


    def backup_fit(self) -> None:
        """Backup fitting method.

        Uses distance transform and if that fails as well, falls back to HoughCircles.
        """
        if self.center_adj_dt():
            self.dt_fail_count = 0
        else:
            pass
            # config.engine.dataout[self.type_entry] = ()
            # self.dt_fail_count += 1
            # if self.dt_fail_count >= self.dt_fail_limit:
            #     # self.logger.info(
            #     #     "DT adjustment failed %d times (exception), falling back to HoughCircles.",
            #     #     self.dt_fail_count,
            #     # )
            #     self.center_adj_hc()
            #     self.dt_fail_count = 0


    def radius_filter(self) -> bool:
        """Filters the radius to avoid sudden jumps."""
        if self.fit_model.params is None:
            self.logger.error("No previous fit parameters, skipping frame.")
            return False

        (filtered_center, new_radius) = self.fit_model.params

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
        try:
            self.fit_model.params = ((float(filtered_center[0]), float(filtered_center[1])), float(filtered_r),)
        except Exception as e:
            self.logger.error(f"2. Error setting fit parameters: {e}")

        self.time_radius_filter = time.perf_counter_ns() / 1e9

        return not is_snap


    def pupil_walkout(self) -> np.ndarray:
        try:
            center = np.round(self.center).astype(int)
            # self.logger.info("Pupil walkout with center: %s", center)
        except Exception:
            raise Exception("No center available for pupil walkout.")

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
        # self.logger.info(crop_list)

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
            np.argmax(canvas_[:, 0][offset_list[0]:] == 0),
            np.argmax(canvas_[0, :][offset_list[1]:] == 0),
            np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][offset_list[2]:] == 0),
            np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[3]:] == 0),
            np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[4]:] == 0),
            np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[5]:] == 0),
            np.argmax(canvas2[-center[1], -center[0]:][offset_list[6]:] == 0),
            np.argmax(canvas2[-center[1]:, -center[0]][offset_list[7]:] == 0),
            np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][offset_list[8]:] == 0),
            np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[9]:] == 0),
            np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[10]:] == 0),
            np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[11]:] == 0),
            np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][offset_list[12]:] == 0),
            np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[13]:] == 0),
            np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[14]:] == 0),
            np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[15]:] == 0),
            np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][offset_list[16]:] == 0),
            np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[17]:] == 0),
            np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[18]:] == 0),
            np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[19]:] == 0),
            np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][offset_list[20]:] == 0),
            np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[21]:] == 0),
            np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[22]:] == 0),
            np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[23]:] == 0),
            np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][offset_list[24]:] == 0),
            np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[25]:] == 0),
            np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[26]:] == 0),
            np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[27]:] == 0),
            np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][offset_list[28]:] == 0),
            np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[29]:] == 0),
            np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[30]:] == 0),
            np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[31]:] == 0)
            ], dtype=int) + offset_list

            # self.logger.info("2. Crop_list sum: %s; threshold: %s", np.sum(crop_list), self.threshold)

            if np.sum(crop_list) < self.threshold:
                # self.logger.warning("Pupil walkout failed: insufficient edge points found.")
                raise IndexError("Pupil walkout failed: crop_list sum: %s !> threshold: %s", np.sum(crop_list), self.threshold)

        # radius = np.sum(crop_list) / (len(crop_stock) * 1.05)
        # self.logger.info("Estimated pupil radius: %.2f", radius)

        r[:8,:] = center
        r[ry_add, 1] += crop_list[ry_add]
        r[rx_add, 0] += crop_list[rx_add]
        r[ry_subtract, 1] -= crop_list[ry_subtract]
        r[rx_subtract, 0] -= crop_list[rx_subtract]
        r[rx_multiplied, 0] *= rx_multiply
        r[ry_multiplied, 1] *= ry_multiply
        r[8:,:] += center

        self.time_walkout = time.perf_counter_ns() / 1e9

        return self.cond(r)


    def cond(self, r) -> np.ndarray:
        dists =  np.linalg.norm(np.mean(r,  axis = 0,dtype=np.float64) - r, axis = 1)

        mean_ = np.mean(dists)
        std_ = np.std(dists)

        lower = mean_ - std_
        upper =  mean_ + std_ * .8
        cond_ = np.logical_and(np.greater_equal(dists, lower), np.less(dists, upper))

        return r[cond_]


    def center_adj_dt(self) -> bool:
        """
        Distance-transform based detection of circular blobs (pupil or CRs).

        Returns True if at least one plausible blob was found, False otherwise.
        """
        if self.type_entry == "cr":
            max_blobs = self.number_of_cr
        else:
            max_blobs = 1

        self.time_center_adj_start = time.perf_counter_ns() / 1e9
        # clear old results
        self.dt_blobs = []

        try:
            if self.source is None:
                return False

            bin_img = self.source  # already binary (0/255) at this point

            # Connected components
            num_labels, labels = cv2.connectedComponents(bin_img)
            if num_labels <= 1:
                # no foreground at all
                # if self.type_entry == "cr":
                    # self.logger.info("DT center_adj: no foreground at all")
                self.time_center_adj_dt = time.perf_counter_ns() / 1e9
                return False

            # previous center guess (if available), used to score candidates
            prev_center: tuple[float, float] | None = None
            if isinstance(self.center, tuple) and len(self.center) == 2:
                prev_center = self.center

            candidates: list[
                tuple[tuple[int, int], float, float, float]
            ] = []  # (center, r_est, circularity, score)

            for lbl in range(1, num_labels):
                # mask for this component
                comp_mask = np.where(labels == lbl, 255, 0).astype(np.uint8)

                area = float(np.sum(comp_mask > 0))
                if area < (self.min_radius ** 2 * np.pi):
                    # too small to be a valid blob
                    # if self.type_entry == "cr":
                        # self.logger.info("DT center_adj: found area too small: %.1f; min_area: %.1f", area, self.min_radius ** 2 * np.pi)
                    continue

                # Distance transform on this component only
                dist = cv2.distanceTransform(
                    comp_mask, cv2.DIST_L2, cv2.DIST_MASK_5
                )
                _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
                r_est = float(maxVal)

                # radius range check
                if r_est < self.min_radius or r_est > self.max_radius:
                    # if self.type_entry == "cr":
                    #     self.logger.info(
                    #         "DT center_adj: found r_est out of range: min_radius: %.1f; r_est: %.1f; max_radius: %.1f",
                    #         self.min_radius,
                    #         r_est,
                    #         self.max_radius,
                    #     )
                    continue

                # circularity check
                expected_area = np.pi * (r_est ** 2)
                if expected_area <= 0:
                    continue
                circularity = area / expected_area

                if not (self.circularity_min <= circularity <= self.circularity_max):
                    # shape too skinny / weird / incomplete
                    # if self.type_entry == "cr":
                    #     self.logger.info("DT center_adj: found circularity out of range: circularity: %.2f; min: %.2f; max: %.2f",
                    #     circularity,
                    #     self.circularity_min,
                    #     self.circularity_max
                    #     )
                    continue

                # aspect ratio check
                ys, xs = np.where(comp_mask > 0)
                h = float(ys.max() - ys.min() + 1)
                w = float(xs.max() - xs.min() + 1)
                aspect = max(h, w) / max(1.0, min(h, w))  # avoid div-by-zero

                if aspect > 2.0:
                    # very elongated blob, probably not a pupil / CR
                    continue

                cx, cy = maxLoc

                # --- Score: radius + circularity + closeness to previous center ---
                mid_r = 0.5 * (self.min_radius + self.max_radius)
                radius_term = abs(r_est - mid_r) / (mid_r + 1e-6)

                # Only use circularity term if the configured range is “tight”
                circ_term = 0.0
                circ_range = self.circularity_max - self.circularity_min
                if circ_range > 1e-3 and circ_range < 10.0:
                    circ_target = 0.5 * (self.circularity_min + self.circularity_max)
                    circ_term = abs(circularity - circ_target) / (circ_target + 1e-6)

                dist_term = 0.0
                if prev_center is not None:
                    dist_term = np.hypot(cx - prev_center[0], cy - prev_center[1]) / (
                        mid_r + 1e-6
                    )
                else:
                    dist_term = 0.0  # no penalty if no previous center

                score = self.w_r * radius_term + self.w_c * circ_term + self.w_d * dist_term

                candidates.append(((cx, cy), r_est, circularity, score))

            if not candidates:
                # nothing passed all checks
                self.time_center_adj_dt = time.perf_counter_ns() / 1e9
                config.engine.dataout[self.type_entry] = ()
                return False

            # sort by score (lower is better)
            candidates.sort(key=lambda c: c[3])

            # keep up to max_blobs best blobs
            top_n = min(max_blobs, len(candidates))
            for i in range(top_n):
                (cx, cy), r_est, circularity, _score = candidates[i]
                self.dt_blobs.append(((float(cx), float(cy)), float(r_est)))

            if self.type_entry == "cr":
                config.engine.dataout[self.type_entry] = self.dt_blobs
            elif self.type_entry == "pupil":
                self.logger.info("center_adj_dt fit success with center: %s.", self.dt_blobs[0][0])
                best_center, best_radius = self.dt_blobs[0]
                self.center = best_center
            else:
                self.logger.error("Unknown type_entry in center_adj_dt: %s", self.type_entry)

            self.time_center_adj_dt = time.perf_counter_ns() / 1e9
            return True

        except Exception as e:
            self.logger.warning("DT center_adj failed with error: %s", e)
            return False


    def center_adj_hc(self) -> None:
        #adjust settings:
        # blurred = cv2.GaussianBlur(self.raw, (3, 3), 2)
        if self.type_entry == "cr":
            self.logger.info("Attempting HoughCircles center adjustment.")
        self.time_center_adj_start = time.perf_counter_ns() / 1e9
        circles = cv2.HoughCircles(
            self.raw,
            cv2.HOUGH_GRADIENT,
            1.5,
            10,
            param1=200,
            param2=15,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is None:
            # self.logger.info("No circles found for center adjustment.")
            return

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
            try:
                self.raw[int(circle[1]), int(circle[0])] = 100
            except IndexError:
                self.logger.warning("Circle index error during center adjustment.")
                self.logger.warning("Raw shape: %s; center: %s", self.raw.shape, circle[:2])

            if smallest == -1:
                smallest = score
                current = circle[:2]
            elif score < smallest:
                smallest = score
                current = circle[:2]
            try:
                if not isinstance(current, np.ndarray) and current != -1:
                    self.center = tuple(current)
                    # self.logger.info("Image shape: %s; Found center: %s", self.raw.shape, self.center)
            except Exception:
                self.logger.error(current)

        self.time_center_adj_hc = time.perf_counter_ns() / 1e9


    def distance(self, a, b) -> float:
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


    def _log_timings(self) -> None:  # type: ignore[operator]
        """Logs the timing information for each processing step."""
        try:
            if self.time_radius_filter is not None:
                self.logger.info(
                    "%s; threshold: %.3fms; walkout=%.3fms, fit=%.3fms, filter=%.3fms, total=%.3fms",
                    self.type_entry,
                    (self.time_threshold - self.time_start) * 1000,  # type: ignore
                    (self.time_walkout - self.time_threshold) * 1000,  # type: ignore
                    (self.time_fit_model - self.time_walkout) * 1000,  # type: ignore
                    (self.time_radius_filter - self.time_fit_model) * 1000,  # type: ignore
                    (self.time_radius_filter - self.time_start) * 1000,  # type: ignore
                )
                pass
            elif self.time_center_adj_dt is not None:
                self.logger.info(
                    "%s; threshold: %.3fms; dt=%.3fms; total=%.3fms",
                    self.type_entry,
                    (self.time_threshold - self.time_start) * 1000,  # type: ignore
                    (self.time_center_adj_dt - self.time_center_adj_start) * 1000,  # type: ignore
                    (self.time_center_adj_dt - self.time_start) * 1000,  # type: ignore
                )
                pass
            elif self.time_center_adj_hc is not None:
                self.logger.info(
                    "%s; threshold: %.3fms; hc=%.3fms; total=%.3fms",
                    self.type_entry,
                    (self.time_threshold - self.time_start) * 1000,  # type: ignore
                    (self.time_center_adj_hc - self.time_center_adj_start) * 1000,  # type: ignore
                    (self.time_center_adj_hc - self.time_start) * 1000,  # type: ignore
                )
                pass
            else:
                self.logger.warning("Incomplete timing information.")
                pass

        except Exception as e:
            self.logger.warning("Timing log error: %s", e)

        self.time_start = self.time_threshold = self.time_walkout = \
            self.time_fit_model = self.time_radius_filter = self.time_center_adj_dt = self.time_center_adj_hc = None
