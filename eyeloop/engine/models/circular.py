# original script: https://github.com/AlliedToasters/circle-fit
# original script author: Michael Klear/AlliedToasters
# hyper-fit doi: https://doi.org/10.1016/j.csda.2010.12.012
# hyper-fit authors: Kenichi Kanatani & Prasanna Rangarajan

import numpy as np
np.seterr('raise')


class Circle:
    def __init__(self, processor) -> None:
        self.shape_processor = processor
        self.fit = self.hyper_fit
        self.params = None

    def hyper_fit(self, r) -> tuple[tuple[float, float], float]:
        """
        Fits coords to circle using hyperfit algorithm.

        Args:
            - coords, list or numpy array with len>2 of the form:
            [
                [x_coord, y_coord],
                ...,
                [x_coord, y_coord]
            ]
            or numpy array of shape (n, 2)
        Returns:
            - ((x_center, y_center), radius)
        """
        X, Y = r[:,0], r[:,1]
        n = X.shape[0]

        mean_X = np.mean(X)
        mean_Y = np.mean(Y)
        Xi = X - mean_X
        Yi = Y - mean_Y
        Xi_sq = Xi**2
        Yi_sq = Yi**2
        Zi = Xi_sq + Yi_sq

        # compute moments

        Mxy = np.sum(Xi * Yi) / n
        Mxx = np.sum(Xi_sq) / n
        Myy = np.sum(Yi_sq) / n
        Mxz = np.sum(Xi * Zi) / n
        Myz = np.sum(Yi * Zi) / n

        Mz = Mxx + Myy

        # finding the root of the characteristic polynomial

        det = (Mxx * Myy - Mxy**2)*2
        #print(det)
        try:
            Xcenter = (Mxz * Myy - Myz * Mxy)/ det
            Ycenter = (Myz * Mxx - Mxz * Mxy)/ det
        except Exception:
            raise IndexError("Error computing x and y center")

        x = float(Xcenter + mean_X)
        y = float(Ycenter + mean_Y)
        r = float(np.sqrt(Xcenter ** 2 + Ycenter ** 2 + Mz))

        self.center = (x, y)

        self.params = (self.center, r)
        #self.center, self.width, self.height, self.angle = self.params

        return self.params
