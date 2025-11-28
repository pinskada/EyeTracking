# ruff: noqa: ERA001, ARG002

"""Simple CR selector based on spatial clustering and basic heuristics."""

import numpy as np
from eyeloop import config

from vr_core.utilities.logger_setup import setup_logger


class SimpleCRSelector:
    """Very simple CR selector on top of DT candidates.

    Idea:
      * We assume that TRUE CRs form a spatial cluster and that after
        your radius/circularity filtering, at most ~half of the candidates
        are false positives (eyelid reflections etc.).
      * We find the densest cluster of candidates within 'cluster_radius'
        and throw away outsiders.
      * Inside that cluster we prefer points that:
          - have many neighbours in the cluster (they're in the middle of it),
          - have reasonable spatial separation from others (Δx/Δy > sep_min_*),
          - optionally are close to previous-frame CR positions.

    This class is intentionally simple and easy to debug; no fancy math.
    """

    def __init__(  # noqa: PLR0913
        self,
        expected_count: int,
        cluster_radius: float = 80.0,
        sep_min_x: float = 5.0,
        sep_min_y: float = 5.0,
        temporal_alpha: float = 0.5,
        temporal_max_dist: float = 20.0,
    ) -> None:
        """Initialize the CR selector.

        Args:
            expected_count:
                Expected number of CRs to select.
            cluster_radius:
                Radius for spatial clustering of candidates.
            sep_min_x:
                Minimum horizontal separation between selected CRs.
            sep_min_y:
                Minimum vertical separation between selected CRs.
            weights:
                Weights for scoring: (neighbor_count, separation, temporal).
            temporal_alpha:
                Alpha for temporal filtering of previous centers.
            temporal_max_dist:
                Maximum distance to previous center to consider it valid.

        """
        self.expected_count = int(expected_count)
        self.cluster_radius = float(cluster_radius)
        self.sep_min_x = float(sep_min_x)
        self.sep_min_y = float(sep_min_y)
        self.temporal_alpha = float(temporal_alpha)
        self.temporal_max_dist = float(temporal_max_dist)
        self.array_width = 2
        # Previous-frame CR centers for simple temporal filtering.
        # List of (x, y) or None if no previous data.
        self.prev_centers: np.ndarray | None = None

        self.eye_side = config.arguments.side
        self.logger = setup_logger(f"{self.eye_side} cr_pattern")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_pattern(
        self,
        candidates: list[tuple[tuple[float, float], float, float, float]],
    ) -> list[tuple[tuple[float, float], float, bool]]:
        """Select CRs from a list of DT candidates.

        The selection process has three steps:
        1. If the side is right eye, flip x-coordinates of candidates.
        2. Finds the densest cluster of candidates and filters outliers.
        3. Ranks candidates inside the cluster using simple consistency scores.
        4. Returns top N candidates according to the scores.

        The rough pattern of the CRs is a half circle around the pupil in the center,
        with new CRs apprering around the pupil as more IR sources are added.
        The pupil centre nor radii is known to this module, so only relative positions
        of the CRs are used. The pattern will be simplified to only left eye as
        the right eye candidates will be flipped.

              Left eye         Right eye

                *         |         *
              *           |           *
                          |
                          |
              *           |           *
                *         |         *

        The general knowledge that the CRs form this pattern is
        implicitly used in the clustering and scoring.

        Args:
            candidates:
                List of tuples:
                    ((cx, cy), radius, circularity, score)

                Only (cx, cy) and radius are used. You can still compute
                'score' upstream however you like (DT quality etc.).

        Returns:
            selected_crs:
                List of:
                    ((cx, cy), radius, is_filled)

        """
        if not candidates:
            self.prev_centers = None
            return []
        self.logger.info(f"Received {len(candidates)} candidates for pattern selection.")
        if self.eye_side == "right":
            # Flip x-coordinates for right eye to simplify pattern logic.
            candidates = [(( -c[0][0], c[0][1]), c[1], c[2], c[3]) for c in candidates]

        # Convert centers and radii to arrays for easier math.
        centers = np.array([c[0] for c in candidates], dtype=np.float32)  # (M, 2)
        radii = np.array([c[1] for c in candidates], dtype=np.float32)    # (M, 1)

        # Find densest cluster of candidates.
        (cluster_centers, cluster_radii, num_candidates) = self._cluster_candidates(  # noqa: RUF059
            centers, radii,
        )

        self.logger.info(f"Clustered to {num_candidates} candidates.")
        return self._pack_cr_list(cluster_centers, cluster_radii)

        # (filtered_centers, filterd_radii, filtered_num_candidates) = self._check_consistency(
        #     cluster_centers, cluster_radii, num_candidates,
        # )

        # cr_list = self._fill_missing_crs(
        #     filtered_centers, filterd_radii, filtered_num_candidates,
        # )


    def _cluster_candidates(
        self,
        centers: np.ndarray[float],
        radii: np.ndarray[float],
    ) -> tuple[np.ndarray[np.dtype[float]], np.ndarray[np.dtype[float]], int]:
        """Cluster candidates and filter obvious outliers.

        The idea:
            1. Compute pairwise distances between all candidates.
            2. For each candidate i, count how many other candidates lie
                within self.cluster_radius (including itself).
            3. Take the candidate with the highest neighbor count as the
                "seed" of the cluster.
            4. Define the cluster as all candidates within self.cluster_radius
                of this seed.

        Args:
            centers:
                Candidate centers, shape (M, 2). Each row is (cx, cy).
            radii:
                Candidate radii, shape (M,) or (M, 1). The shape is not
                changed here, only subsetted by the cluster indices.

        Returns:
            cluster_centers:
                Centers of candidates belonging to the densest cluster,
                shape (K, 2), where K is the cluster size.
            cluster_radii:
                Radii of candidates belonging to the densest cluster,
                shape (K,) or (K, 1) matching the input shape.
            num_candidates:
                Number of candidates in the cluster (K). If there were no
                input candidates, K = 0 and the returned arrays are empty.

        """
        # No candidates at all -> return empty cluster.
        if centers.size == 0:
            empty_centers = centers.reshape(0, 2)
            empty_radii = radii.reshape(0, *radii.shape[1:])
            return empty_centers, empty_radii, 0

        # Ensure we have (M, 2) shape for centers.
        centers = np.asarray(centers, dtype=np.float32)
        max_dim_shape = 2
        if centers.ndim != max_dim_shape or centers.shape[1] != max_dim_shape:
            exception_msg = (f"centers must have shape (M, 2), got {centers.shape}")
            raise ValueError(exception_msg)

        # Pairwise differences: diff[i, j] = centers[i] - centers[j]
        diff = centers[:, None, :] - centers[None, :, :]  # (M, M, 2)

        # Euclidean distances between all pairs.
        dists = np.linalg.norm(diff, axis=2)  # (M, M)

        # For each candidate i, count neighbors within cluster_radius
        # (including itself).
        neighbor_counts = np.sum(dists <= self.cluster_radius, axis=1)  # (M,)

        # Index of the candidate with the highest neighbor count.
        seed_idx = int(np.argmax(neighbor_counts))

        # Indices of candidates belonging to the same cluster as the seed.
        cluster_mask = dists[seed_idx] <= self.cluster_radius
        cluster_indices = np.nonzero(cluster_mask)[0]

        # Subset centers and radii to get the cluster arrays.
        cluster_centers = centers[cluster_indices]
        cluster_radii = radii[cluster_indices]
        num_candidates = int(cluster_centers.shape[0])

        return cluster_centers, cluster_radii, num_candidates


    def _check_consistency(
        self,
        cluster_centers: np.ndarray[float],
        cluster_radii: np.ndarray[float],
        num_candidates: int,
    ) -> tuple[np.ndarray[float], np.ndarray[float], int]:
        """Check the relative pattern of the remaining CRs and filter obious outliers.

        1. Sorts candidates by vertical coordinate.
        2. Computes relative distances and angles between candidates based on the expected pattern.
        3. Filters candidates that deviate significantly from the expected pattern.

        No temporal filtering is done here.

        The current implementation is deprecated, complete new version will be done.

        Args:
            cluster_centers:
                Candidate centers (N, 2).
            cluster_radii:
                Candidate radii (N, 1).
            num_candidates:
                Number of candidates (N).

        Returns:
            filtered_centers:
                Centers that passed the consistency check.
            filterd_radii:
                Radii that passed the consistency check.
            filtered_count:
                Number of candidates that passed the consistency check.

        """
        return ([0.1], [0.1], 0)  # Placeholder implementation.


    def _fill_missing_crs(
        self,
        cluster_centers: np.ndarray[float],
        cluster_radii: np.ndarray[float],
        num_candidates: int,
    ) -> list[tuple[float, float], float, bool]:
        """Fill missing CRs based on the expected pattern.

        Based on the expected pattern of CRs and maybe temporal information, we can estimate positions
        of missing CRs and fill them in. This can help maintain a consistent
        number of CRs across frames, which will be crucial for calculating CR centroid.

        This is a placeholder for future implementation.

        Args:
            cluster_centers:
                Candidate centers (N, 2).
            cluster_radii:
                Candidate radii (N, 1).
            num_candidates:
                Number of candidates (N).

        Returns:
            list of:
                ((cx, cy), radius, is_filled)

        """
        # Placeholder implementation: return empty list.
        return []


    def _pack_cr_list(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        is_filled: np.ndarray | None = None,
    ) -> list[tuple[tuple[float, float], float, bool]]:
        """Convert internal array representation back to a list of CR tuples.

        This is useful for testing intermediate steps: you can run one
        step (e.g. clustering), then immediately return its reduced set
        of CRs in the same format as create_pattern() would use.

        Args:
            centers:
                Array of CR centers, shape (K, 2): [[cx0, cy0], [cx1, cy1], ...].

            radii:
                Array of CR radii, shape (K, 1).

            is_filled:
                Optional boolean array of shape (K, 1) indicating whether
                each CR was synthetically filled (True) or actually
                detected (False). If None, all CRs are marked as
                detected (False).

        Returns:
            List of:
                ((cx, cy), radius, is_filled)
            in image coordinates. For the right eye, x is un-flipped
            back to the original coordinate system.

        """
        # No CRs -> empty list
        if centers is None or centers.size == 0:
            return []

        # Ensure proper shapes/dtypes
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != self.array_width or centers.shape[1] != self.array_width:
            exception_msg = (f"centers must have shape (K, 2), got {centers.shape}")
            raise ValueError(exception_msg)

        radii = np.asarray(radii, dtype=np.float32).reshape(-1)
        if radii.shape[0] != centers.shape[0]:
            exception_msg = (f"radii length ({radii.shape[0]}) does not match "
                             f"centers length ({centers.shape[0]})")
            raise ValueError(exception_msg)
        if is_filled is None:
            filled = np.zeros(centers.shape[0], dtype=bool)
        else:
            filled = np.asarray(is_filled, dtype=bool).reshape(-1)
            if filled.shape[0] != centers.shape[0]:
                exception_msg = (
                    f"is_filled length ({filled.shape[0]}) does not match "
                    f"centers length ({centers.shape[0]})"
                )
                raise ValueError(exception_msg)

        # Un-flip x for right eye so external users always see real image coords
        out_centers = centers.copy()
        if self.eye_side == "right":
            out_centers[:, 0] *= -1.0

        # Build output list
        cr_list: list[tuple[tuple[float, float], float, bool]] = []
        for (cx, cy), r, f in zip(out_centers, radii, filled, strict=True):
            cr_list.append(((float(cx), float(cy)), float(r), bool(f)))

        return cr_list
