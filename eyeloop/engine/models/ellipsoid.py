from typing import Optional


import numpy as np
np.seterr('raise')


class Ellipse:
    def __init__(self, processor):
        self.shape_processor = processor
        self.params: Optional[tuple[tuple[float, float], float, float, float]] = None
        self.coef: Optional[np.ndarray] = None

    def fit(self, r: np.ndarray):
        """
        Least-squares ellipse fit (Halir & Flusser) implemented with ndarrays.

        r : (N, 2) ndarray of edge points (x, y)
        """
        # ---- NO try/except here: let caller handle failures ----

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
        phi = 0.5 * np.arctan2(2.0 * b, ac_subtr)
        angle_deg = float(np.rad2deg(phi) % 360.0)

        self.params = ((float(x0), float(y0)),
                       float(width),
                       float(height),
                       angle_deg)

        return self.params[0]
