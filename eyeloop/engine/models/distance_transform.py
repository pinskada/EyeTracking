# ruff: noqa: ERA001

"""Distance transform based blob detection for pupil and corneal reflections."""
import time

import cv2
import numpy as np
from eyeloop import config
from eyeloop.engine.models.cr_pattern_tracker import SimpleCRSelector

from vr_core.eye_tracker import tracker_types as tt
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
        self.w_r, self.w_c, self.w_d = weights
        self.number_of_cr = number_of_points if number_of_points is not None else 1

        # NOTE: for CR tracking the original code never updated self.center,
        # so it typically stays -1 and prev_center is None. We keep that
        # behaviour for compatibility.
        self.center: tuple[float, float] | int = -1

        self.eye_side = config.arguments.side
        self.logger = setup_logger(f"{self.eye_side} DistTransform")

        # timing (filled in detect)
        self.time_center_adj_start: float | None = None
        self.time_center_adj_dt: float | None = None

        # list of detected blobs: [((cx, cy), r_est), ...]
        self.dt_blobs: list[tuple[tuple[float, float], float]] = []
        self.height = 0

        self.cr_pattern_tracker = SimpleCRSelector(expected_count=self.number_of_cr)

    def detect(  # noqa: C901
        self,
        source: np.ndarray,
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
            if source is None:
                return None

            bin_img = source  # already binary (0/255) at this point
            offset_x: int = 0
            if self.track_type == "cr":
                if self.eye_side == "Left":
                    # If side is left, select only left half of the image
                    bin_img = bin_img[:, : bin_img.shape[1] // 2]
                else:
                    # If side is right, select only right half of the image
                    offset_x = bin_img.shape[1] // 2
                    bin_img = bin_img[:, bin_img.shape[1] // 2 :]
            self.height = bin_img.shape[0]
            # Connected components with stats (area + bounding box)
            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                bin_img
            )
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

            candidates = self._filter_blobs(
                num_labels,
                labels,
                stats,
                prev_center,
            )
            if not candidates:
                # nothing passed all checks
                self.time_center_adj_dt = time.perf_counter_ns() / 1e9
                config.engine.dataout[self.track_type] = ()
                return None

            if self.track_type == "cr":
                self._process_crs(candidates, offset_x, max_blobs)
            elif self.track_type == "pupil":
                self._process_pupil(candidates)
            else:
                self.logger.error(
                    "Unknown track_type in center_adj_dt: %s", self.track_type
                )

            self.time_center_adj_dt = time.perf_counter_ns() / 1e9

            # NOTE: original code returns self.center. For CR this is usually -1;
            # for pupil it is updated to the best blob center.
            return self.center  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            self.logger.warning("DT center_adj failed with error for %s: %s", self.track_type, e)
            return None


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
        LEFT = cv2.CC_STAT_LEFT
        TOP = cv2.CC_STAT_TOP
        WIDTH = cv2.CC_STAT_WIDTH
        HEIGHT = cv2.CC_STAT_HEIGHT
        AREA = cv2.CC_STAT_AREA

        for lbl in range(1, num_labels):
            # --- Basic geometric properties from stats ---
            x = int(stats[lbl, LEFT])
            y = int(stats[lbl, TOP])
            w = int(stats[lbl, WIDTH])
            h = int(stats[lbl, HEIGHT])
            area = float(stats[lbl, AREA])

            # if self.track_type == "cr" and self.eye_side == "Left" and y >= self.height // 2:
            #     self.logger.info(
            #         "Area: %f", area
            #     )
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

            # if self.track_type == "cr" and self.eye_side == "Left" and y >= self.height // 2:
            #     self.logger.info("Estimated radius: %f", r_est)
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
            # if self.track_type == "cr" and self.eye_side == "Left" and y >= self.height // 2:
            #     self.logger.info("Circularity: %f", circularity)
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

            # if self.track_type == "cr" and self.eye_side == "Left" and y >= self.height // 2:
            #     self.logger.info("Aspect ratio: %f", aspect)
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
        offset_x: int,
        max_blobs: int,
    ) -> None:
        """Process CR candidates to select the best ones."""
        # filtered_candidates = self.cr_pattern_tracker.create_pattern(candidates, offset_x)
        # config.engine.dataout[self.track_type] = filtered_candidates

        candidates.sort(key=lambda c: c.score)
        candidates = candidates[:max_blobs]
        try:
            for i in range(len(candidates)):
                center = candidates[i].center
                radius = candidates[i].radius_estimate

                center = (center[0] + offset_x, center[1])

                # Convert from cropped coordinates to full-image coordinates
                self.dt_blobs.append(
                    tt.CrData(center, radius, False)
                )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("CR processing failed with error: %s", e)
        config.engine.dataout[self.track_type] = self.dt_blobs
