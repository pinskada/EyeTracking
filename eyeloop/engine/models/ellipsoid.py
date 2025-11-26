from typing import Optional


import numpy as np
np.seterr('raise')


class Ellipse:
    def __init__(self, processor):
        self.shape_processor = processor
        self.params: Optional[tuple[tuple[float, float], float]] = None
        self.coef: Optional[np.ndarray] = None

    def fit(self, r: np.ndarray) -> tuple[tuple[float, float], float]:
        """
        Least-squares ellipse fit (Halir & Flusser) implemented with ndarrays.

        Args:
            r: (N, 2) ndarray of edge points (x, y)
        Returns:
            Center of the fitted ellipse (x0, y0)
        """
        # Ensure ndarray + float64 for numerical stability
        r = np.asarray(r, dtype=np.float64)
        x = r[:, 0]
        y = r[:, 1]

        if r.shape[0] < 5:
            raise ValueError("Need at least 5 points to fit ellipse")

        # ------------------------------------------------------------------
        # Design matrices (Halir & Flusser, eq. 15–16)
        # ------------------------------------------------------------------
        D1 = np.column_stack((x * x, x * y, y * y))          # (N, 3)
        D2 = np.column_stack((x, y, np.ones_like(x)))        # (N, 3)

        # Scatter matrices (eq. 17)
        S1 = D1.T @ D1                                       # (3, 3)
        S2 = D1.T @ D2                                       # (3, 3)
        S3 = D2.T @ D2                                       # (3, 3)

        # Constraint matrix (eq. 18)
        C1 = np.array(
            [[0.0, 0.0, 2.0],
             [0.0, -1.0, 0.0],
             [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        # ------------------------------------------------------------------
        # Reduced scatter matrix (eq. 29)
        # ------------------------------------------------------------------
        S3_inv = np.linalg.inv(S3)
        C1_inv = np.linalg.inv(C1)

        M = C1_inv @ (S1 - S2 @ S3_inv @ S2.T)

        # ------------------------------------------------------------------
        # Solve eigenproblem M * a1 = λ * a1 for quadratic part a1=[a,b,c]^T
        # ------------------------------------------------------------------
        eigvals, eigvecs = np.linalg.eig(M)  # (3,), (3,3)

        # Enforce ellipse constraint 4ac - b^2 > 0, with tolerance
        a = eigvecs[0, :]
        b = eigvecs[1, :]
        c = eigvecs[2, :]
        cond = 4.0 * a * c - b * b

        # Allow small negative values due to numerical noise
        idx = np.where(cond > 1e-8)[0]
        if idx.size == 0:
            # Fall back to the eigenvector with largest cond
            idx = [int(np.argmax(cond))]

        a1 = eigvecs[:, idx[0:1]]          # (3,1)

        # Linear part a2 = [d, f, g]^T (eq. 24)
        a2 = -S3_inv @ S2.T @ a1           # (3,1)

        # Full parameter vector [a, b, c, d, f, g]^T (6,1)
        self.coef = np.vstack((a1, a2))

        # ------------------------------------------------------------------
        # Convert algebraic parameters to geometric params
        # General conic: a x^2 + 2b x y + c y^2 + 2d x + 2f y + g = 0
        # ------------------------------------------------------------------
        a = float(self.coef[0, 0])
        b = float(self.coef[1, 0]) / 2.0
        c = float(self.coef[2, 0])
        d = float(self.coef[3, 0]) / 2.0
        f = float(self.coef[4, 0]) / 2.0
        g = float(self.coef[5, 0])

        # Center (x0, y0) (eq. 19–20)
        af = a * f
        cd = c * d
        bd = b * d
        ac = a * c
        b_sq = b * b
        z_ = (b_sq - ac)

        if abs(z_) < 1e-12:
            raise ValueError("Degenerate conic (z_≈0)")

        x0 = (cd - b * f) / z_
        y0 = (af - bd) / z_

        # Semi-axes (eq. 21–22)
        ac_subtr = a - c
        if abs(ac_subtr) < 1e-12:
            raise ValueError("Degenerate conic (a≈c)")

        numerator = 2.0 * (af * f + cd * d + g * b_sq - 2.0 * bd * f - ac * g)
        denom = ac_subtr * np.sqrt(1.0 + 4.0 * b_sq / (ac_subtr ** 2))

        denominator1 = ( -denom - c - a) * z_
        denominator2 = (  denom - c - a) * z_

        if denominator1 == 0.0 or denominator2 == 0.0:
            raise ValueError("Degenerate ellipse (denominator≈0)")

        width  = np.sqrt(numerator / denominator1)
        height = np.sqrt(numerator / denominator2)

        # Rotation angle in degrees
        # phi = 0.5 * np.arctan2(2.0 * b, ac_subtr)
        # angle_deg = float(np.rad2deg(phi) % 360.0)

        # self.params = ((float(x0), float(y0)),
        #                float(width),
        #                float(height),
        #                angle_deg)
        aproximate_radius = (float(width) + float(height)) / 2.0
        center = (float(x0), float(y0))
        self.params = (center, aproximate_radius)

        return self.params


class Fast_Elliptical_Stable:
    """
    Faster, center-focused ellipse fit with temporal smoothing.

    - Linear LS fit of a conic:
        A x^2 + B x y + C y^2 + D x + E y + F = 0, F = -1
    - Extracts center (cx, cy) from 2x2 linear system.
    - Estimates radius from distances to center.
    - Uses previous params to smooth out jumps.

    Output:
        params = ((cx, cy), r_est)
    """

    def __init__(self):
        # params = ((cx, cy), r_est)
        self.params: tuple[tuple[float, float], float] | None = None

        # Smoothing / clamp hyperparams (tune!)
        self.alpha_center = 0.5   # 0..1, 1 = no smoothing
        self.alpha_radius = 0.5   # 0..1
        self.max_center_jump = 3.0  # px per frame
        self.max_radius_jump = 3.0  # px per frame

    def fit(self, r) -> tuple[tuple[float, float], float]:
        r = np.asarray(r, dtype=np.float64)
        if r.ndim != 2 or r.shape[1] != 2:
            raise ValueError("Fast_Elliptical.fit expects (N, 2) array")
        n_pts = r.shape[0]
        if n_pts < 5:
            # Not enough points -> crude fallback: mean + median radius
            center = np.mean(r, axis=0)
            dists = np.linalg.norm(r - center, axis=1)
            r_est = float(np.median(dists)) if dists.size else 0.0
            center = (float(center[0]), float(center[1]))
            self._update_params(center, r_est)
            return self.params

        x = r[:, 0]
        y = r[:, 1]

        # Shift coordinates to improve conditioning
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x0 = x - x_mean
        y0 = y - y_mean

        # Linear system for A,B,C,D,E with F fixed at -1:
        # A x^2 + B x y + C y^2 + D x + E y - 1 = 0
        A_mat = np.column_stack((x0 * x0, x0 * y0, y0 * y0, x0, y0))  # (N, 5)
        b_vec = np.ones_like(x0)

        try:
            p, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            A_c, B_c, C_c, D_c, E_c = p
        except np.linalg.LinAlgError:
            # Fallback: no model, just keep previous or mean
            center = np.mean(r, axis=0)
            dists = np.linalg.norm(r - center, axis=1)
            r_est = float(np.median(dists)) if dists.size else 0.0
            center = (float(center[0]), float(center[1]))
            self._update_params(center, r_est)
            return self.params

        # Check ellipse-ish condition: 4AC - B^2 > 0
        cond = 4.0 * A_c * C_c - B_c * B_c
        if cond <= 1e-8:
            # Degenerate conic (parabola/hyperbola) -> don't trust it, use crude
            center = np.mean(r, axis=0)
            dists = np.linalg.norm(r - center, axis=1)
            r_est = float(np.median(dists)) if dists.size else 0.0
            center = (float(center[0]), float(center[1]))
            self._update_params(center, r_est)
            return self.params

        # Solve for center in shifted coordinates:
        # [2A  B ] [cx0] = [-D]
        # [ B 2C] [cy0]   [-E]
        M = np.array([[2.0 * A_c, B_c],
                      [B_c,       2.0 * C_c]], dtype=np.float64)
        rhs = np.array([-D_c, -E_c], dtype=np.float64)
        try:
            cx0, cy0 = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            cx0, cy0 = 0.0, 0.0

        # Shift back to original coordinates
        cx = float(cx0 + x_mean)
        cy = float(cy0 + y_mean)
        center = np.array([cx, cy], dtype=np.float64)

        # Raw radius estimate
        dists = np.linalg.norm(r - center, axis=1)
        if dists.size == 0:
            r_raw = 0.0
        elif dists.size < 5:
            r_raw = float(np.mean(dists))
        else:
            med = float(np.median(dists))
            mad = float(np.median(np.abs(dists - med))) + 1e-6
            mask = np.abs(dists - med) < 2.5 * mad
            r_raw = float(np.mean(dists[mask])) if np.any(mask) else med

        # Temporal smoothing + jump limiting
        # center_smooth, r_smooth = self._smooth(center, r_raw)
        center_smooth = center
        r_smooth = r_raw
        self.params = ( (float(center_smooth[0]), float(center_smooth[1])),
                        float(r_smooth) )
        return self.params

    def _smooth(self, center_raw: np.ndarray, r_raw: float):
        """
        Smooth against previous params and clamp jumps.
        """
        if self.params is None:
            return center_raw, r_raw

        prev_center, prev_r = self.params
        prev_center = np.asarray(prev_center, dtype=np.float64)
        prev_r = float(prev_r)

        # Clamp center jump
        delta_c = center_raw - prev_center
        dist = float(np.linalg.norm(delta_c))
        if dist > self.max_center_jump:
            if dist > 1e-6:
                delta_c *= self.max_center_jump / dist
            center_raw = prev_center + delta_c

        # Clamp radius jump
        delta_r = r_raw - prev_r
        if abs(delta_r) > self.max_radius_jump:
            delta_r = np.sign(delta_r) * self.max_radius_jump
            r_raw = prev_r + delta_r

        # Exponential smoothing
        c_alpha = self.alpha_center
        r_alpha = self.alpha_radius

        center_smooth = (1.0 - c_alpha) * prev_center + c_alpha * center_raw
        r_smooth = (1.0 - r_alpha) * prev_r + r_alpha * r_raw

        return center_smooth, r_smooth

    def _update_params(self, center: tuple[float, float], r_est: float):
        """
        Helper for crude fallbacks: still apply temporal smoothing if we have history.
        """
        center_arr = np.array(center, dtype=np.float64)
        center_smooth, r_smooth = self._smooth(center_arr, r_est)
        self.params = ( (float(center_smooth[0]), float(center_smooth[1])),
                        float(r_smooth) )
