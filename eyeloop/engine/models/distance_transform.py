# ruff: noqa: ERA001

"""Distance transform based blob detection for pupil and corneal reflections."""
import time

import cv2
import numpy as np
from eyeloop import config

from vr_core.eye_tracker import tracker_types as tt
from vr_core.eye_tracker.eyeloop_module.eyeloop.engine.models.cr_pattern_tracker import CrPatternTracker
from vr_core.utilities.logger_setup import setup_logger


class DistanceTransform:
    """Distance transform based blob detection for pupil and corneal reflections."""

    def __init__(  # noqa: PLR0913
        self,
        track_type: str,
        min_radius: float,
        max_radius: float,
        circularity_min: float,
        circularity_max: float,
        aspect_ratio_max: float,
        mask_radius: float | None = None,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        number_of_points: int | None = None,
    ) -> None:
        """Initialize DistanceTransform parameters."""
        self.track_type = track_type
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.circularity_min = circularity_min
        self.circularity_max = circularity_max
        self.aspect_ratio_max = aspect_ratio_max
        self.mask_radius = mask_radius
        self.w_r, self.w_c, self.w_d = weights
        self.number_of_cr = number_of_points if number_of_points is not None else 1

        self.mask_angle_start_deg = 110.0
        self.mask_angle_end_deg = 360.0

        self.center: tuple[float, float] | int = -1

        self.eye_side = config.arguments.side
        self.logger = setup_logger(f"{self.eye_side} DistTransform")

        # timing (filled in detect)
        self.time_center_adj_start: float | None = None
        self.time_center_adj_dt: float | None = None

        # list of detected blobs: [((cx, cy), r_est), ...]
        self.dt_blobs: list[tuple[tuple[float, float], float]] = []
        self.height = 0

        self.cr_pattern_tracker = CrPatternTracker(side=self.eye_side, num_crs=self.number_of_cr)

    def detect(
        self,
        bin_img: np.ndarray,
    ) -> tuple[float, float] | None:
        """Distance-transform based detection of circular blobs (pupil or CRs).

        Matches the behaviour of the original implementation but
        uses connectedComponentsWithStats and per-blob ROIs for speed.
        """
        max_blobs = self.number_of_cr if self.track_type == "cr" else 1

        self.time_center_adj_start = time.perf_counter_ns() / 1e9
        # clear old results
        self.dt_blobs = []

        try:
            if bin_img is None:
                return None

            if self.track_type == "cr":
                # self.logger.info("DT center_adj: masking for CRs.")
                pupil_center = self._get_pupil_center()
                bin_img = self._mask_circle(bin_img, pupil_center)
            shape = bin_img.shape
            # Connected components with stats (area + bounding box)
            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                bin_img,
            )
            config.engine.cr_processor.source = bin_img
            if num_labels <= 1:
                # no foreground at all
                # if self.track_type == "cr":
                #     self.logger.info("DT center_adj: no foreground at all")
                self.time_center_adj_dt = time.perf_counter_ns() / 1e9
                return None

            # previous center guess (if available), used to score candidates
            prev_center: tuple[float, float] | None = None
            center_length = 2
            if isinstance(self.center, tuple) and len(self.center) == center_length:
                prev_center = self.center

            try:
                candidates = self._filter_blobs(
                    num_labels,
                    labels,
                    stats,
                    prev_center,
                )
            except Exception as e:  # noqa: BLE001
                self.logger.info(
                    "DT center_adj: blob filtering failed with error for %s: %s",
                    self.track_type,
                    e,
                )
                return None
            if not candidates:
                # nothing passed all checks
                self.time_center_adj_dt = time.perf_counter_ns() / 1e9
                config.engine.dataout[self.track_type] = ()
                return None

            if self.track_type == "cr":
                self._process_crs(candidates, pupil_center, shape, max_blobs)
            elif self.track_type == "pupil":
                self._process_pupil(candidates)
            else:
                self.logger.error(
                    "Unknown track_type in center_adj_dt: %s", self.track_type,
                )

            self.time_center_adj_dt = time.perf_counter_ns() / 1e9

            # NOTE: original code returns self.center. For CR this is usually -1;
            # for pupil it is updated to the best blob center.
            return self.center  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            self.logger.info("DT center_adj failed with error for %s: %s", self.track_type, e)
            return None


    def _get_pupil_center(self) -> tuple[float, float] | None:
        """Get the current pupil center from dataout."""
        pupil_data = config.engine.dataout.get("pupil")
        pupil_center = getattr(pupil_data, "center", None) if pupil_data else None
        if pupil_center is None or len(pupil_center) != 2:  # noqa: PLR2004
            error = "Pupil center is not available for masking."
            raise ValueError(error)

        px, py = float(pupil_center[0]), float(pupil_center[1])
        return (px, py)


    def _mask_circle(
            self,
            bin_img: np.ndarray,
            pupil_center: tuple[float, float],
        ) -> np.ndarray:
        """Mask out everything outside a (possibly sector-limited) circle around the pupil.

        Uses OpenCV drawing ops (circle + filled polygon) instead of per-pixel trig,
        which is much faster than the previous numpy-based implementation.
        """
        # No masking configured → return as-is
        if self.mask_radius is None or self.mask_radius <= 0:
            return bin_img

        h, w = bin_img.shape[:2]
        px, py = pupil_center

        # Clamp pupil center into image bounds for safety
        cx = int(np.clip(px, 0, w - 1))
        cy = int(np.clip(py, 0, h - 1))
        radius = int(self.mask_radius)

        # --- 1) Prepare angle range in degrees (0-360) ---

        start_deg = float(self.mask_angle_start_deg)
        end_deg = float(self.mask_angle_end_deg)

        # Mirror horizontally for RIGHT eye (same logic as before)
        # angle' = 360° - angle
        if self.eye_side == "Right":
            start_deg, end_deg = (360.0 - end_deg) % 360.0, (360.0 - start_deg) % 360.0

        # Normalize to [0, 360)
        start_deg = start_deg % 360.0
        end_deg = end_deg % 360.0

        # If the sector covers (almost) full circle, just draw a disc
        sweep = (end_deg - start_deg) % 360.0
        full_circle = sweep >= 359.0 or np.isclose(sweep, 0.0, atol=1e-2)

        # --- 2) Build mask image ---

        mask = np.zeros((h, w), dtype=np.uint8)

        if full_circle:
            # Simple filled circle around the pupil
            cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
        else:
            def draw_sector(s_deg: float, e_deg: float) -> None:
                """Draw a filled circular sector from s_deg to e_deg."""
                # Ensure angles are in ascending order
                s_deg = s_deg % 360.0
                e_deg = e_deg % 360.0

                # If still wrapping (e < s), shift end by +360 for linspace
                if e_deg < s_deg:
                    e_deg += 360.0

                # Step ~5° along the arc (at least a few points)
                n_steps = max(8, int((e_deg - s_deg) / 5.0))
                angles_deg = np.linspace(s_deg, e_deg, n_steps, dtype=np.float32)
                angles_rad = np.deg2rad(angles_deg)

                # Our convention:
                #   0°   = up    (0, -1)
                #   90°  = right (1,  0)
                #   180° = down  (0,  1)
                #   270° = left  (-1, 0)
                # matches:
                #   x = cx + r * sin(theta)
                #   y = cy - r * cos(theta)
                xs = cx + radius * np.sin(angles_rad)
                ys = cy - radius * np.cos(angles_rad)

                # Build polygon: center -> arc -> back to center
                pts = np.stack(
                    [
                        np.concatenate([[cx], xs, [cx]]),
                        np.concatenate([[cy], ys, [cy]]),
                    ],
                    axis=1,
                )
                pts = np.round(pts).astype(np.int32)

                cv2.fillConvexPoly(mask, pts, 255)

            # Non-wrapping sector (start <= end): draw once
            if start_deg <= end_deg:
                draw_sector(start_deg, end_deg)
            else:
                # Wrapping sector (e.g. 300° → 60°): split into [start, 360] U [0, end]
                draw_sector(start_deg, 360.0)
                draw_sector(0.0, end_deg)

        # --- 3) Apply mask to binary image ---

        # mask is single-channel; this works for both 1- and 3-channel bin_img
        return cv2.bitwise_and(bin_img, bin_img, mask=mask)


    def _mask_circle0(
        self,
        bin_img: np.ndarray,
        pupil_center: tuple[float, float],
    ) -> np.ndarray:
        """Mask out everything outside a circle centered at self.center."""
        if self.mask_radius is None or self.mask_radius <= 0:
            return bin_img

        px, py = pupil_center

        h, w = bin_img.shape[:2]

        # Clamp pupil center to valid range (safety)
        cx = float(np.clip(px, 0, w - 1))
        cy = float(np.clip(py, 0, h - 1))

        # ----- 1) Build coordinate grid relative to pupil center -----
        yy, xx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        dx = xx - cx
        dy = yy - cy

        # Radius
        r = np.sqrt(dx * dx + dy * dy)

        # ----- 2) Screen-space angles -----
        # We want:
        #   0°   at "up"    (0, -1)
        #   90°  at "right" (1,  0)
        #   180° at "down"  (0,  1)
        #   270° at "left"  (-1, 0)
        #
        # With image coords (y down), this is achieved by:
        #   theta = atan2(dx, -dy)
        theta = np.arctan2(dx, -dy)  # range [-pi, pi]
        two_pi = 2.0 * np.pi
        theta = (theta + two_pi) % two_pi  # -> [0, 2*pi)

        # ----- 3) Base sector for LEFT eye -----
        start_deg = float(self.mask_angle_start_deg)
        end_deg = float(self.mask_angle_end_deg)

        # Mirror horizontally for RIGHT eye:
        #   theta' = 180° - theta  (mod 360)
        # So interval [start, end] becomes [180-end, 180-start].
        if self.eye_side == "Right":
            # Horizontal mirror across vertical axis: angle' = 360° - angle
            start_deg, end_deg = (360.0 - end_deg) % 360.0, (360.0 - start_deg) % 360.0

        a_start = np.deg2rad(start_deg) % two_pi
        a_end = np.deg2rad(end_deg) % two_pi

        # ----- 4) Angular mask with wrap-around support -----
        ang_mask = (theta >= a_start) & (theta <= a_end) if a_start <= a_end else (theta >= a_start) | (theta <= a_end)

        # ----- 5) Radial mask (disc; if you want a ring, add inner radius) -----
        rad_mask = r <= float(self.mask_radius)

        mask = ang_mask & rad_mask

        out = bin_img.copy()
        out[~mask] = 0
        # out[mask] = 250

        return out

    def _filter_blobs(
        self,
        num_labels: int,
        labels: np.ndarray,
        stats: np.ndarray,
        prev_center: tuple[float, float] | None,
    ) -> list[tt.DTCandidate]:
        """Filter connected components to find plausible blobs.

        Returns a list of (center, r_est, circularity, score).
        """
        candidates: list[
            tt.DTCandidate
        ] = []  # (center, r_est, circularity, score)

        # shorthand for OpenCV stat indices
        left = cv2.CC_STAT_LEFT
        top = cv2.CC_STAT_TOP
        width = cv2.CC_STAT_WIDTH
        height = cv2.CC_STAT_HEIGHT
        area_cv = cv2.CC_STAT_AREA
        for lbl in range(1, num_labels):
            # --- Basic geometric properties from stats ---
            x = int(stats[lbl, left])
            y = int(stats[lbl, top])
            w = int(stats[lbl, width])
            h = int(stats[lbl, height])
            area = float(stats[lbl, area_cv])

            # area check (too small to be a valid blob)
            if area < (self.min_radius ** 2 * np.pi):
                # if self.track_type == "cr":
                #     self.logger.info(
                #         "DT center_adj: found area too small: %.1f; min_area: %.1f",
                #         area,
                #         self.min_radius ** 2 * np.pi,
                #     )
                continue

            # ROI mask for this component only (avoids full-image np.where per label)
            roi_labels = labels[y : y + h, x : x + w]
            comp_mask = (roi_labels == lbl).astype(np.uint8)

            # Distance transform on this component only
            dist = cv2.distanceTransform(
                comp_mask,
                cv2.DIST_L2,
                cv2.DIST_MASK_5,
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(dist)
            r_est = float(max_val)

            # radius range check
            if r_est < self.min_radius or r_est > self.max_radius:
                # if self.track_type == "cr":
                #     self.logger.info(
                #         "DT center_adj: found r_est out of range: min_radius: "
                #         "%.1f; r_est: %.1f; max_radius: %.1f",
                #         self.min_radius,
                #         r_est,
                #         self.max_radius,
                #   )
                continue

            # circularity check
            expected_area = np.pi * (r_est**2)
            if expected_area <= 0:
                continue
            circularity = area / expected_area

            if not (self.circularity_min <= circularity <= self.circularity_max):
                # shape too skinny / weird / incomplete
                # if self.track_type == "cr":
                #     self.logger.info(
                #         "DT center_adj: found circularity out of "
                #         "range: circularity: %.2f; min: %.2f; max: %.2f",
                #         circularity,
                #         self.circularity_min,
                #         self.circularity_max,
                #     )
                continue

            # aspect ratio check (using bbox; matches ys/xs max-min+1 logic)
            h_f = float(h)
            w_f = float(w)
            aspect = max(h_f, w_f) / max(1.0, min(h_f, w_f))  # avoid div-by-zero

            if aspect > self.aspect_ratio_max:
                # very elongated blob, probably not a pupil / CR
                continue

            # blob center in full (cropped) coordinates
            cx = x + max_loc[0]
            cy = y + max_loc[1]

            # --- Score: radius + circularity + closeness to previous center ---
            mid_r = 0.5 * (self.min_radius + self.max_radius)
            radius_term = abs(r_est - mid_r) / (mid_r + 1e-6)

            # Only use circularity term if the configured range is “tight”
            circ_term = 0.0
            circ_range = self.circularity_max - self.circularity_min
            if 1e-3 < circ_range < 10.0:  # noqa: PLR2004
                circ_target = 0.5 * (self.circularity_min + self.circularity_max)
                circ_term = abs(circularity - circ_target) / (circ_target + 1e-6)

            if prev_center is not None:
                dist_term = np.hypot(cx - prev_center[0], cy - prev_center[1]) / (
                    mid_r + 1e-6
                )
            else:
                dist_term = 0.0  # no penalty if no previous center

            score = self.w_r * radius_term + self.w_c * circ_term + self.w_d * dist_term

            candidates.append(tt.DTCandidate((cx, cy), r_est, area, circularity, score))

        return candidates


    def _process_pupil(
        self,
        candidates: list[tt.DTCandidate],
    ) -> None:
        """Process pupil candidates to select the best one."""
        candidates.sort(key=lambda c: c.score)
        pupil = candidates[0]

        self.center = pupil.center


    def _process_crs(
        self,
        candidates: list[tt.DTCandidate],
        pupil_center: tuple[float, float],
        shape: tuple[int, int],
        max_blobs: int,
    ) -> None:
        """Process CR candidates to select the best ones."""
        candidates.sort(key=lambda c: c.score)
        candidates = candidates[:max_blobs]

        # filtered_candidates = self.cr_pattern_tracker.update_candidates(candidates, pupil_center, shape)

        # config.engine.dataout[self.track_type] = filtered_candidates

        # self.logger.info("Outputting %d CR candidates.", len(candidates))
        try:
            for i in range(len(candidates)):
                center = candidates[i].center
                radius = candidates[i].radius_estimate

                center = (center[0], center[1])

                # Convert from cropped coordinates to full-image coordinates
                self.dt_blobs.append(
                    tt.CrData(center, radius, False),
                )
        except Exception as e:
            self.logger.warning("CR processing failed with error: %s", e)
        config.engine.dataout[self.track_type] = self.dt_blobs
