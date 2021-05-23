from typing import Optional

import numpy as np

from distributions.abstract import AbstractDistribution


class Gaussian(AbstractDistribution):
    """
    Gaussian (Normal) distributions with parameters mu and sigma.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        if y is None:
            self.mu = X.mean()
            self.sigma = X.std()
        else:
            n_classes = max(y) + 1
            self.mu = np.zeros(n_classes)
            self.sigma = np.zeros(n_classes)

            for cls in range(n_classes):
                self.mu[cls] = X[y == cls].mean()
                self.sigma[cls] = X[y == cls].std()
