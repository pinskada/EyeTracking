"""CR pattern tracker."""

from __future__ import annotations

import math
from dataclasses import dataclass

import vr_core.eye_tracker.tracker_types as tt
from vr_core.utilities.logger_setup import setup_logger


@dataclass
class _CrSlot:
    """Internal representation of one CR in the pattern (in flipped coords).

    All geometry is stored in a coordinate system where the *pattern side*
    is the same for both eyes. For the Left eye, x is flipped so that
    the CRs always appear on the same side of the pupil in this internal
    space.

    Attributes:
        center_flipped:
            (x, y) center of the CR in the flipped coordinate space.
        angle:
            Angle of the CR relative to the pupil center, in radians,
            in the flipped coordinate space. Normalized to [0, 2π).
        radius:
            Distance of the CR from the pupil center, in pixels.
        missing_count:
            How many consecutive frames this slot has been estimated
            (i.e. no direct detection). Can be used later to discard
            very stale estimates.

    """

    center_flipped: tuple[float, float]
    angle: float
    radius: float
    missing_count: int = 0


class CrPatternTracker:
    """Cr pattern tracker class.

    - Maintains an angularly ordered buffer of CR positions.
    - Matches new DTCandidate detections to buffer slots by angle
      (and loosely by radius).
    - For slots with no detection in a frame, estimates a likely
      position based on the average motion of the detected CRs.
    - Returns a list of CrData where estimated CRs are marked
      with ``is_filled = True``.
    """

    def __init__(
        self,
        side: str,
        num_crs: int = 6,
    ) -> None:
        """Initialize the CR pattern tracker.

        Args:
            side: Eye side identifier.
            num_crs: Number of corneal reflections to track.

        """
        if side not in ("Left", "Right"):
            self.logger.error("CrPatternTracker side should be 'Left' or 'Right', got %s", side)

        self.logger = setup_logger(f"CrPatternTracker_{side}")

        self.side = side
        self.num_crs = num_crs

        # Internal buffer: angularly ordered slots describing the CR pattern.
        self._slots: list[_CrSlot] = []
        self._initialized: bool = False

        # Store last pupil center in flipped coordinates so we can
        # recompute CR centers from angle + radius.
        self._last_pupil_center_flipped: tuple[float, float] | None = None

        # Image width from the last frame (needed for flipping).
        self._last_width: int | None = None

        # Simple matching / estimation parameters.
        # These are intentionally conservative and can be tuned later.
        self.angle_threshold_rad: float = math.radians(25.0)
        self.radius_rel_threshold: float = 0.5  # 50 % change allowed
        self.min_detections_for_update: int = max(2, num_crs // 2)
        self.max_missing_frames: int = 5  # currently not enforced aggressively


    # ##################### Public API #####################

    def update_candidates(
        self,
        candidates: list[tt.DTCandidate],
        pupil_center: tuple[float, float],
        shape: tuple[int, int],
    ) -> list[tt.CrData]:
        """Update CR candidates based on proximity to pupil center.

        Args:
            candidates:
                List of detected CR candidates.
            pupil_center:
                Tuple of (x, y) coordinates of the pupil center.
            shape:
                Size of the eye image (width, height).

        Returns:
            List of filtered CR data.

        """
        # 1. Flip x coordinates of Left eye pupil and crs to make the algorithm eye side consistent
        # 2. Assume pupil center as a rough center of the circle formed by the CRs (the pupil center relative to
        #    the crs positions will change slightly between frames) -> compute their angles and distances
        # 3. If first run, save num_crs of crs to a buffer in angle increasing order, or wait several frames
        #    to fill num_crs candidates
        # 4. For each new frame, compute angles and distances of candidates to pupil center
        # 5. Match new candidates to crs in the buffer based on angle relative angle
        # 6. Update buffer with new cr positions
        # 7. If a cr is not present in the new candidates, estimate its new position based on previous position
        #    and average movement of other crs
        # 8. Flip back x coordinates of Left eye cr before returning
        # 9. Return updated a list[tt.CrData] of crs

        # If only half or less (0-3/6) of the required CRs are detected, return empty list

        _, width = shape  # height unused, but keeps signature explicit
        self._last_width = width

        if not candidates:
            # No detections at all this frame.
            if not self._initialized or not self._slots:
                # Nothing to estimate from yet - fall back to empty list.
                return []
            # Reuse previous pattern, recomputed around current pupil center,
            # and mark all as filled.
            self._last_pupil_center_flipped = self._flip_point(pupil_center, width)
            estimated_indices = list(range(len(self._slots)))
            self._recompute_slot_centers()
            return self._build_output(estimated_indices)

        # Convert candidates into polar coordinates (angle & radius)
        # in flipped space, relative to the pupil center.
        polar_candidates, pupil_center_flipped = self._compute_angles_distances(
            candidates,
            pupil_center,
            width,
        )
        self._last_pupil_center_flipped = pupil_center_flipped

        # If we are not initialized yet, try to initialize the pattern.
        if not self._initialized:
            self._try_initialize_slots(polar_candidates)
            if not self._initialized:
                # Not enough information yet - just pass through raw detections.
                return self._create_cr_data_list(candidates)

        # Match polar candidates to existing slots.
        (
            matched_indices,
            angle_deltas,
            radius_deltas,
        ) = self._match_candidates_to_buffer(polar_candidates)

        num_matches = len(matched_indices)
        if num_matches < self.min_detections_for_update:
            # Too few reliable detections to estimate a meaningful motion.
            # We still return the buffered pattern, but mark everything as filled.
            self._recompute_slot_centers()
            estimated_indices = list(range(len(self._slots)))
            return self._build_output(estimated_indices)

        # Estimate positions for slots without a match in this frame.
        estimated_indices = self._estimate_missing_crs(
            matched_indices,
            angle_deltas,
            radius_deltas,
        )

        # Build final CrData list; estimated ones are flagged.
        return self._build_output(estimated_indices)


    # ##################### Internal helpers #####################

    def _flip_x_coord(self, x: float, width: int) -> float:
        """Flip x-coordinate for the Left eye so that pattern side matches.

        Right eye: x is returned unchanged.
        Left eye:  x is mirrored around the vertical center line.
        """
        if self.side == "Left":
            return float(width - 1 - x)
        return float(x)


    def _flip_point(self, pt: tuple[float, float], width: int) -> tuple[float, float]:
        """Flip a 2D point in x if needed for the Left eye."""
        x, y = pt
        return self._flip_x_coord(x, width), float(y)


    def _compute_angles_distances(
        self,
        candidates: list[tt.DTCandidate],
        pupil_center: tuple[float, float],
        width: int,
    ) -> tuple[list[dict], tuple[float, float]]:
        """Compute angle and radius of each candidate in flipped space.

        Returns
        -------
        polar_candidates:
            List of dicts with keys:
            - "candidate": original DTCandidate
            - "center_flipped": (x, y) center in flipped coords
            - "angle": angle in radians in [0, 2π)
            - "radius": radial distance from pupil center
        pupil_center_flipped:
            Pupil center in flipped coordinates.

        """
        px, py = pupil_center
        px_f = self._flip_x_coord(px, width)
        pupil_center_flipped = (px_f, float(py))

        polar_candidates: list[dict] = []
        for cand in candidates:
            cx, cy = cand.center
            cx_f = self._flip_x_coord(cx, width)
            dx = cx_f - px_f
            dy = cy - py

            angle = math.atan2(dy, dx)
            if angle < 0.0:
                angle += 2.0 * math.pi
            radius = math.hypot(dx, dy)

            polar_candidates.append(
                {
                    "candidate": cand,
                    "center_flipped": (cx_f, float(cy)),
                    "angle": angle,
                    "radius": radius,
                },
            )

        return polar_candidates, pupil_center_flipped


    def _try_initialize_slots(self, polar_candidates: list[dict]) -> None:
        """Initialize angularly ordered CR slots from the first good frame.

        The pattern is initialized once, given enough plausible candidates.
        """
        if len(polar_candidates) <= self.num_crs // 2:
            # Not enough information yet; wait for a better frame.
            return

        # Sort by angle and take up to num_crs.
        polar_candidates_sorted = sorted(polar_candidates, key=lambda c: c["angle"])
        selected = polar_candidates_sorted[: self.num_crs]

        self._slots = [
            _CrSlot(
                center_flipped=pc["center_flipped"],
                angle=pc["angle"],
                radius=pc["radius"],
                missing_count=0,
            )
            for pc in selected
        ]
        self._initialized = True
        self.logger.debug("Initialized CR pattern with %d slots", len(self._slots))


    def _match_candidates_to_buffer(
        self,
        polar_candidates: list[dict],
    ) -> tuple[dict[int, int], list[float], list[float]]:
        """Match polar candidates to existing slots by angle (and loosely radius).

        Returns:
            matched_indices:
                Mapping slot_index -> polar_candidate_index that was matched.
            angle_deltas:
                List of (new_angle - previous_angle) for matched slots.
            radius_deltas:
                List of (new_radius - previous_radius) for matched slots.

        """
        matched_indices: dict[int, int] = {}
        used_candidate_indices: set[int] = set()
        angle_deltas: list[float] = []
        radius_deltas: list[float] = []

        for slot_idx, slot in enumerate(self._slots):
            best_idx: int | None = None
            best_angle_diff = float("inf")

            for cand_idx, pc in enumerate(polar_candidates):
                if cand_idx in used_candidate_indices:
                    continue

                # Angular difference on a circle.
                diff = abs(pc["angle"] - slot.angle)
                if diff > math.pi:
                    diff = 2.0 * math.pi - diff

                if diff > self.angle_threshold_rad:
                    continue

                radius_diff = abs(pc["radius"] - slot.radius)
                if slot.radius > 0 and radius_diff > self.radius_rel_threshold * slot.radius:
                    continue

                if diff < best_angle_diff:
                    best_angle_diff = diff
                    best_idx = cand_idx

            if best_idx is not None:
                used_candidate_indices.add(best_idx)
                matched_indices[slot_idx] = best_idx

                pc = polar_candidates[best_idx]
                angle_deltas.append(pc["angle"] - slot.angle)
                radius_deltas.append(pc["radius"] - slot.radius)

                # Update slot with new detection.
                slot.center_flipped = pc["center_flipped"]
                slot.angle = pc["angle"]
                slot.radius = pc["radius"]
                slot.missing_count = 0  # reset missing streak

        return matched_indices, angle_deltas, radius_deltas


    def _estimate_missing_crs(
        self,
        matched_indices: dict[int, int],
        angle_deltas: list[float],
        radius_deltas: list[float],
    ) -> list[int]:
        """Estimate positions of missing CRs based on previous positions.

        For simplicity, we apply the average angular and radial motion of
        the matched CRs to all missing slots.

        Returns:
            estimated_indices:
                Indices of slots that were *estimated* in this frame (and should
                be output with ``is_filled=True``).

        """
        estimated_indices: list[int] = []

        if self._last_pupil_center_flipped is None:
            return estimated_indices

        # Compute average motion from matched CRs.
        mean_d_angle = sum(angle_deltas) / len(angle_deltas) if angle_deltas else 0.0

        mean_d_radius = sum(radius_deltas) / len(radius_deltas) if radius_deltas else 0.0

        # Update missing slots using the average motion.
        for idx, slot in enumerate(self._slots):
            if idx in matched_indices:
                # Already updated from a real detection.
                continue

            slot.angle += mean_d_angle
            # Keep angle in [0, 2π).
            slot.angle %= 2.0 * math.pi
            slot.radius += mean_d_radius
            slot.radius = max(slot.radius, 0.0)  # prevent negative radius

            slot.missing_count += 1
            if slot.missing_count > self.max_missing_frames:
                # Currently we just keep using it; later you could drop or
                # reinitialize the pattern here.
                pass

            estimated_indices.append(idx)

        # Recompute centers for all slots around the current pupil center.
        self._recompute_slot_centers()
        return estimated_indices


    def _recompute_slot_centers(self) -> None:
        """Recompute slot.center_flipped from angle + radius around pupil."""
        if self._last_pupil_center_flipped is None:
            return

        px, py = self._last_pupil_center_flipped
        for slot in self._slots:
            dx = slot.radius * math.cos(slot.angle)
            dy = slot.radius * math.sin(slot.angle)
            slot.center_flipped = (px + dx, py + dy)


    def _create_cr_data_list(
        self,
        candidates: list[tt.DTCandidate],
    ) -> list[tt.CrData]:
        """Create CrData from DTCandidate without any pattern tracking.

        All CRs produced here are direct detections, so ``is_filled`` is
        always False.
        """
        cr_list: list[tt.CrData] = [
            tt.CrData(
                center=cand.center,
                radius=cand.radius_estimate,
                is_filled=False,
            )
            for cand in candidates
        ]
        return cr_list


    def _build_output(self, estimated_indices: list[int]) -> list[tt.CrData]:
        """Convert internal slots to CrData list with correct is_filled flags."""
        if self._last_width is None:
            # Should not normally happen, but be defensive.
            return []

        estimated_set = set(estimated_indices)
        cr_list: list[tt.CrData] = []

        for idx, slot in enumerate(self._slots):
            x_f, y = slot.center_flipped
            x = self._flip_x_coord(x_f, self._last_width)
            is_filled = idx in estimated_set
            cr_list.append(tt.CrData(center=(x, y), radius=slot.radius, is_filled=is_filled))

        return cr_list
