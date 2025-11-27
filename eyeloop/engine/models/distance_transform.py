# ruff: noqa: ERA001

"""Distance transform based blob detection for pupil and corneal reflections."""
import time

import cv2
import numpy as np
from eyeloop import config

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

        self.center = -1

        self.eye_side = config.arguments.side
        self.logger = setup_logger(f"{self.eye_side} DistTransform")

    def detect(  # noqa: C901
            self,
            source: np.ndarray,
    ) -> tuple[float, float] | None:
            """Distance-transform based detection of circular blobs (pupil or CRs).

            Returns True if at least one plausible blob was found, False otherwise.
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

                # Connected components
                num_labels, labels = cv2.connectedComponents(bin_img)
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
                    prev_center,
                )

                if not candidates:
                    # nothing passed all checks
                    self.time_center_adj_dt = time.perf_counter_ns() / 1e9
                    config.engine.dataout[self.track_type] = ()
                    return ()

                # sort by score (lower is better)
                candidates.sort(key=lambda c: c[3])

                # keep up to max_blobs best blobs
                top_n = min(max_blobs, len(candidates))
                for i in range(top_n):
                    (cx, cy), r_est, circularity, _score = candidates[i]  # noqa: RUF059
                    self.dt_blobs.append(((float(cx + offset_x), float(cy)), float(r_est)))
                    # if self.track_type == "pupil":
                    #     self.logger.info("Radius: %f, Circularity: %f", r_est, circularity)

                if self.track_type == "cr":
                    config.engine.dataout[self.track_type] = self.dt_blobs
                elif self.track_type == "pupil":
                    # self.logger.info("center_adj_dt fit success with center: %s.", self.dt_blobs[0])
                    best_center, _ = self.dt_blobs[0]
                    self.center = best_center
                else:
                    self.logger.error("Unknown track_type in center_adj_dt: %s", self.track_type)

                self.time_center_adj_dt = time.perf_counter_ns() / 1e9

                return self.center  # noqa: TRY300

            except Exception as e:  # noqa: BLE001
                self.logger.warning("DT center_adj failed with error: %s", e)
                return None


    def _filter_blobs(
        self,
        num_labels: int,
        labels: np.ndarray,
        prev_center: tuple[float, float] | None,
    ) -> list[tuple[tuple[int, int], float, float, float]]:
        """Filter connected components to find plausible CRs."""
        candidates: list[
            tuple[tuple[int, int], float, float, float]
        ] = []  # (center, r_est, circularity, score)

        for lbl in range(1, num_labels):
            # mask for this component
            comp_mask = np.where(labels == lbl, 255, 0).astype(np.uint8)

            area = float(np.sum(comp_mask > 0))
            if area < (self.min_radius ** 2 * np.pi):
                # too small to be a valid blob
                # if self.track_type == "cr":
                    # self.logger.info("DT center_adj: found area too small: %.1f; min_area: %.1f",
                    # area,
                    # self.min_radius ** 2 * np.pi
                #)
                continue

            # Distance transform on this component only
            dist = cv2.distanceTransform(
                comp_mask, cv2.DIST_L2, cv2.DIST_MASK_5,
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(dist)
            r_est = float(max_val)

            # radius range check
            if r_est < self.min_radius or r_est > self.max_radius:
                # if self.track_type == "cr":
                #     self.logger.info(
                #         "DT center_adj: found r_est out of range: min_radius: \
                #           %.1f; r_est: %.1f; max_radius: %.1f",
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
                # if self.track_type == "cr":
                #     self.logger.info("DT center_adj: found circularity out of \
                #        range: circularity: %.2f; min: %.2f; max: %.2f",
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

            if aspect > self.aspect_ratio_max:
                # very elongated blob, probably not a pupil / CR
                continue

            cx, cy = max_loc

            # --- Score: radius + circularity + closeness to previous center ---
            mid_r = 0.5 * (self.min_radius + self.max_radius)
            radius_term = abs(r_est - mid_r) / (mid_r + 1e-6)

            # Only use circularity term if the configured range is “tight”
            circ_term = 0.0
            circ_range = self.circularity_max - self.circularity_min
            if circ_range > 1e-3 and circ_range < 10.0:  # noqa: PLR2004
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

        return candidates
