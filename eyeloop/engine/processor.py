# ruff: noqa: F403, F405, ERA001, TRY400, TRY002

"""Processor module for eye features (pupil, corneal reflection)."""

import collections
import time

import cv2
import numpy as np
from eyeloop import config
from eyeloop.constants.processor_constants import *
from eyeloop.engine.models.circular import Circle
from eyeloop.engine.models.distance_transform import DistanceTransform
from eyeloop.engine.models.ellipsoid import Ellipse, Fast_Elliptical_Stable

from vr_core.utilities.logger_setup import setup_logger


class Shape:
    """Shape processor for eye features (pupil, corneal reflection)."""

    def __init__(self, track_type: str) -> None:
        """Initialize the Shape processor."""
        self.side = config.arguments.side

        track_type_pupil = "pupil"
        track_type_cr = "cr"

        self.process_blink = False

        self.logger = setup_logger(f"{self.side} processor")
        self.active = False
        self.center = -1

        self.filtered_radius = None
        self.filtered_center = None

        self.walkout_offset = 0

        self.track_type = track_type

        self.side = config.arguments.side

        self.brightness_threshold = 5.0

        self.dt_fail_count = 0
        self.dt_fail_limit = 3

        if track_type == track_type_pupil:
            self._setup_pupil_params()
        elif track_type == track_type_cr:
            self._setup_cr_params()
        else:
            self.logger.error("Unknown processor track_type: %s", track_type)

        self.last_min_radius = self.min_radius
        self.compute_threshold()

        self.time_start: float | None = None
        self.time_threshold: float | None = None
        self.time_walkout: float | None = None
        self.time_fit_model: float | None = None
        self.time_dt_fit_start: float | None = None
        self.time_dt_fit_end: float | None = None
        self.time_radius_filter: float | None = None


    def _setup_pupil_params(self) -> None:
        """Set up pupil processor parameters."""
        # Threshold settings ----------------------------------------------
        self.binarythreshold = -1 # Binary threshold (computed later)
        self.blur = (3, 3) # Blur size for thresholding

        # Radius settings -------------------------------------------------
        self.min_radius = 2 # Minimum expected radius for pupil detection
        self.max_radius = 100 # Maximum expected radius for pupil detection

        # Distance transform settings ------------------------------------
        self.circularity_min = 1.0 # Minimum circularity for pupil detection
        self.circularity_max = 5 # Maximum circularity for pupil detection
        self.aspect_ratio_max = 2 # Minimum aspect ratio for pupil detection

        self.w_r = 0.4   # Score weight for radius
        self.w_c = 0.4   # Score weight for circularity
        self.w_d = 0.5   # Score weight for distance to previous center

        self.distance_transform = DistanceTransform(
            self.track_type,
            self.min_radius,
            self.max_radius,
            self.circularity_min,
            self.circularity_max,
            self.aspect_ratio_max,
            weights=(self.w_r, self.w_c, self.w_d),
        )

        # Radius filter settings -----------------------------------------
        self.radius_drop_factor = 0.0 # Maximum drop factor for radius in one frame
        radius_buffer_size = 20 # Size of the buffer for radius filtering

        self.radius_buffer = collections.deque(maxlen=radius_buffer_size)

        model = config.arguments.model

        if model == "circular":
            self.fit_model = Circle(self)
        elif model == "elliptical":
            self.fit_model = Ellipse(self)
        elif model == "fast_elliptical":
            self.fit_model = Fast_Elliptical_Stable()
        else:
            self.logger.error("Unknown model: %s", model)

        self.apply_thresh = self.pupil_thresh_


    def _setup_cr_params(self) -> None:
        """Set up corneal reflection processor parameters."""
        self.number_of_cr = 4 # Set how many CRs to track

        # Threshold settings ----------------------------------------------
        self.binarythreshold = 200 # Binary threshold
        self.blur = (1, 1) # Blur size for thresholding

        # Radius settings -------------------------------------------------
        self.min_radius = 1 # Minimum expected radius for CR detection
        self.max_radius = 5 # Maximum expected radius for CR detection

        # Distance transform settings -------------------------------------
        self.circularity_min = 1 # Minimum circularity for pupil detection
        self.circularity_max = 2 # Maximum circularity for pupil detection
        self.aspect_ratio_max = 2 # Minimum aspect ratio for pupil detection

        self.w_r = 0.4   # Score weight for radius
        self.w_c = 0.4   # Score weight for circularity
        self.w_d = 0.2   # Score weight for distance to previous center

        self.apply_thresh = self.cr_thresh_

        self.distance_transform = DistanceTransform(
            self.track_type,
            self.min_radius,
            self.max_radius,
            self.circularity_min,
            self.circularity_max,
            self.aspect_ratio_max,
            weights=(self.w_r, self.w_c, self.w_d),
        )


    def compute_threshold(self) -> None:
        """Compute the threshold for pupil detection based on min_radius."""
        self.threshold = len(crop_stock) * self.min_radius * 1.05
        self.logger.info(
            "Type: %s_%s; Min_radius: %s; Max_radius: %s; Threshold: %s",
            config.arguments.side, self.track_type, self.min_radius,
            self.max_radius, self.threshold,
        )


    def track(self, source: np.ndarray) -> None:
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

        if self.process_blink:
            try:
                config.blink[config.blink_i] = mean_img
                config.blink_i += 1

            except IndexError:
                config.blink_i = 0

            baseline = np.mean(config.blink[np.nonzero(config.blink)])
            diff = np.abs(mean_img - baseline)

            # self.logger.info("Mean image intensity: %.2f, baseline: %.2f, diff: %.2f", mean_img, baseline, diff)

            if diff > self.brightness_threshold:
                config.engine.dataout[self.track_type] = ()
                self.logger.info("Blink detected.")
                return
        if self.track_type == "pupil":
            self.pupil_fit()
        else:
            self.time_dt_fit_start = time.perf_counter_ns() / 1e9
            center = self.distance_transform.detect(self.source)
            self.time_dt_fit_end = time.perf_counter_ns() / 1e9
            if center is not None:
                self.center = center

        # self._log_timings()


    def pupil_thresh_(self) -> None:
        """Apply pupil blur and thresholding to the source image."""
        self.source[:] = cv2.threshold(
            cv2.GaussianBlur(
                cv2.erode(self.source, kernel, iterations = 1),
                self.blur,
                0,
            ),
            self.binarythreshold,
            255,
            cv2.THRESH_BINARY_INV,
        )[1]


    def cr_thresh_(self) -> None:
        """Apply cr blur and thresholding to the source image."""
        _, self.source[:] = cv2.threshold(
            cv2.GaussianBlur(self.source, self.blur, 0),
            self.binarythreshold,
            255,
            cv2.THRESH_BINARY,
        )


    def pupil_fit(self) -> None:
        """Fit the pupil model to the detected boundary points."""
        try:
            r = self.pupil_walkout()
            self.time_walkout = time.perf_counter_ns() / 1e9
            fit_params = self.fit_model.fit(r)
            # self.logger.info("Pupil fit success.")
            self.time_fit_model = time.perf_counter_ns() / 1e9

            self.center = fit_params[0]

            # raw_r = (self.fit_model.params[1] + self.fit_model.params[2]) / 2.0
            try:
                # frame_valid = self.radius_filter()
                frame_valid = True
            except Exception as e:  # noqa: BLE001
                self.logger.error("Radius filter error: %s", e)
                frame_valid = True

            if self.track_type is not None and self.fit_model.params is not None:
                if frame_valid:
                    # Normal tracking output
                    config.engine.dataout[self.track_type] = fit_params
                else:
                    # Snap: return empty output
                    config.engine.dataout[self.track_type] = ()
            # self.logger.info("Pupil radius: %.2f", self.fit_model.params[1])
            # if config.arguments.side == "Right":
            #     self.logger.info("raw=%.3f filtered=%.3f", raw_r, self.fit_model.params[1])


        # If pupil_walkout or fit fails, fall back to distance transform
        except IndexError:
            # self.logger.info(f"Fit index error: {e}")
            self.time_dt_fit_start = time.perf_counter_ns() / 1e9
            center = self.distance_transform.detect(self.source)
            self.time_dt_fit_end = time.perf_counter_ns() / 1e9
            if center is not None:
                self.center = center

        except Exception:  # noqa: BLE001
            # self.logger.info(f"Fit-func error: {e}")
            self.time_dt_fit_start = time.perf_counter_ns() / 1e9
            center = self.distance_transform.detect(self.source)
            self.time_dt_fit_end = time.perf_counter_ns() / 1e9
            if center is not None:
                self.center = center


    def radius_filter(self) -> bool:
        """Filter the radius to avoid sudden jumps."""
        if self.fit_model.params is None:
            self.logger.error("No previous fit parameters, skipping frame.")
            return False

        (filtered_center, new_radius) = self.fit_model.params

        self.radius_buffer.append(new_radius)

        is_snap = False

        # If we have at least 3 previous samples, compare against their mean
        min_prev_samples = 3
        if len(self.radius_buffer) > min_prev_samples:
            # Mean of previous radii (exclude the newest one)
            prev_radii = list(self.radius_buffer)[:-1]
            mean_radius = float(np.mean(prev_radii))

            if new_radius < mean_radius * self.radius_drop_factor:
                is_snap = True

                # Suspiciously small radius: KEEP last filtered radius in the output
                filtered_r = self.filtered_radius if self.filtered_radius is not None else mean_radius

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
            self.fit_model.params = ((float(filtered_center[0]), float(filtered_center[1])), float(filtered_r))
        except Exception as e:  # noqa: BLE001
            self.logger.error("2. Error setting fit parameters: %s", e)

        #self.time_radius_filter = time.perf_counter_ns() / 1e9

        return not is_snap


    def pupil_walkout(self) -> np.ndarray:  # noqa: C901, PLR0915
        """Radial walkout from the current center on a cleaned pupil mask.

        - Fills small CR holes and artifacts inside the pupil.
        - Uses a local ROI around the center for speed.
        - Casts multiple rays and finds the pupil boundary as the first
        255 -> 0 transition along each ray.
        - Returns boundary points filtered by `cond()`.
        """
        if self.center is None or not isinstance(self.center, (tuple, list, np.ndarray)):
            except_msg = "No center available for pupil walkout."
            raise Exception(except_msg)

        cx, cy = np.round(self.center).astype(int)

        canvas = self.source
        if canvas is None:
            except_msg = "No source image for pupil walkout."
            raise Exception(except_msg)

        h, w = canvas.shape[:2]

        # Clamp center to valid range (avoid out-of-bounds)
        cx = int(np.clip(cx, 1, w - 2))
        cy = int(np.clip(cy, 1, h - 2))

        # ---------- 1) Build local ROI around center ----------
        # Margin just a bit larger than max_radius
        margin = int(self.max_radius + 4)

        x0 = max(cx - margin, 0)
        x1 = min(cx + margin, w - 1)
        y0 = max(cy - margin, 0)
        y1 = min(cy + margin, h - 1)

        roi = canvas[y0:y1 + 1, x0:x1 + 1]

        # Local center coords inside ROI
        rcx = cx - x0
        rcy = cy - y0

        # ---------- 2) Keep only largest pupil component & fill holes ----------
        # Pupil pixels are 255 after THRESH_BINARY_INV
        max_white = 255
        pupil_mask = (roi == max_white).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            pupil_mask, connectivity=8,
        )

        if num_labels <= 1:
            # No foreground at all inside ROI
            except_msg = "pupil_walkout: no foreground in pupil ROI"
            raise IndexError(except_msg)

        # Find largest non-background component (label > 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(1 + np.argmax(areas))

        cleaned = (labels == largest_label).astype(np.uint8)

        # Morphological closing to fill small holes (CRs) inside pupil
        # Kernel size 3x3 is usually enough; you can tune to (5,5) if CRs are bigger
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        # Re-binarize to 0/255
        cleaned = (cleaned * 255).astype(np.uint8)

        # ---------- 3) Make sure center lies inside cleaned pupil ----------
        if cleaned[rcy, rcx] == 0:
            ys, xs = np.where(cleaned > 0)
            if xs.size == 0:
                except_msg = "pupil_walkout: cleaned pupil is empty"
                raise IndexError(except_msg)

            # Snap center to nearest pupil pixel
            d2 = (xs - rcx) ** 2 + (ys - rcy) ** 2
            idx = int(np.argmin(d2))
            rcx, rcy = int(xs[idx]), int(ys[idx])

        # ---------- 4) Radial ray casting ----------
        # Number of rays: tuneable. 32 is a good compromise.
        n_rays = 32
        angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)

        min_r = max(1, int(self.min_radius))
        max_r = int(self.max_radius)

        points = []

        for theta in angles:
            dx = float(np.cos(theta))
            dy = float(np.sin(theta))

            hit = None
            inside_seen = False

            # Walk from min_radius to max_radius along this ray
            for r in range(min_r, max_r):
                x = round(rcx + dx * r)
                y = round(rcy + dy * r)

                if x <= 0 or x >= (x1 - x0) or y <= 0 or y >= (y1 - y0):
                    break

                val = cleaned[y, x]

                if val == max_white:
                    # We're inside pupil
                    inside_seen = True
                    continue

                # We were inside pupil and now hit 0 -> boundary
                if inside_seen and val == 0:
                    hit = (x, y)
                    break

            if hit is not None:
                # Convert back to full-image coordinates
                gx = hit[0] + x0
                gy = hit[1] + y0
                points.append((gx, gy))

        min_rays_required = 10
        if len(points) < min_rays_required:
            # Not enough valid boundary samples -> let dt backup handle it
            except_msg = f"pupil_walkout: only {len(points)} valid rays found"
            raise IndexError(except_msg)

        r = np.asarray(points, dtype=np.float64)
        mean_r = np.mean(r, axis=0)  # noqa: F841
        # self.logger.info(r)

        return self.cond(r)


    def cond(self, r: np.ndarray) -> np.ndarray:
        """Robustly filter boundary points based on distance from their mean center.

        Uses median + MAD (median absolute deviation) to remove outliers,
        which makes it much more tolerant to a few bad rays (e.g. from eyelashes
        or remaining artifacts).
        """
        array_length = 2
        r = np.asarray(r, dtype=np.float64)
        if r.ndim != array_length or r.shape[1] != array_length:
            except_msg = "cond expects (N, 2) array"
            raise ValueError(except_msg)

        min_points = 5
        if r.shape[0] < min_points:
            # Too few points to meaningfully filter; just return as-is
            return r

        center = np.mean(r, axis=0)
        dists = np.linalg.norm(r - center, axis=1)

        med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - med))) + 1e-6  # avoid div-by-zero

        # z-score using MAD; 0.6745 scales MAD to ~std for Gaussian
        z = 0.6745 * (dists - med) / mad

        # Keep points within |z| < 2.5 (tunable)
        z_threshold = 2.5
        mask = np.abs(z) < z_threshold

        r_filt = r[mask]

        # If filtering kills too many points, fall back to original
        if r_filt.shape[0] < max(8, r.shape[0] // 3):
            return r

        return r_filt


    def _log_timings(self) -> None:  # track_type: ignore[operator]
        """Log the timing information for each processing step."""
        try:
            if self.time_fit_model is not None:
                self.logger.info(
                    "%s; Threshold: %.3fms; Walkout=%.3fms, Fit=%.3fms, Total=%.3fms; FPS=%.2f",
                    self.track_type,
                    (self.time_threshold - self.time_start) * 1000,  # track_type: ignore
                    (self.time_walkout - self.time_threshold) * 1000,  # track_type: ignore
                    (self.time_fit_model - self.time_walkout) * 1000,  # track_type: ignore
                    (self.time_fit_model - self.time_start) * 1000,  # track_type: ignore
                    1 / (self.time_fit_model - self.time_start),  # track_type: ignore
                )
            elif self.time_dt_fit_end is not None:
                self.logger.info(
                    "%s; Threshold: %.3fms; dt=%.3fms; total=%.3fms; FPS=%.2f",
                    self.track_type,
                    (self.time_threshold - self.time_start) * 1000,  # track_type: ignore
                    (self.time_dt_fit_end - self.time_dt_fit_start) * 1000,  # track_type: ignore
                    (self.time_dt_fit_end - self.time_start) * 1000,  # track_type: ignore
                    1 / (self.time_dt_fit_end - self.time_start),  # track_type: ignore
                )
            else:
                self.logger.warning("Incomplete timing information.")

        except Exception as e:  # noqa: BLE001
            self.logger.warning("Timing log error: %s", e)

        self.time_start = self.time_threshold = self.time_walkout = \
            self.time_fit_model = self.time_dt_fit_start = self.time_dt_fit_end = None
