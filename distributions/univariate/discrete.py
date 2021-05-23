from typing import Optional

import numpy as np

from distributions.abstract import AbstractDistribution


class Bernoulli(AbstractDistribution):
    """
    Bernoulli distributions with parameter prob.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.prob = X.mean()
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = X[y == cls].mean()
