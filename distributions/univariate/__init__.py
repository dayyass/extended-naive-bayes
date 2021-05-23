from typing import Optional

import numpy as np
from scipy.stats import rv_continuous

from distributions.abstract import AbstractDistribution
from distributions.univariate.continuous import (  # noqa: F401
    Beta,
    Exponential,
    Gamma,
    Gaussian,
)
from distributions.univariate.discrete import (  # noqa: F401
    Bernoulli,
    Binomial,
    Categorical,
    Geometric,
    Poisson,
)


class ContinuousUnivariateDistribution(AbstractDistribution):
    """
    Any continuous univariate distribution from scipy.stats with method "fit"
    (scipy.stats.rv_continuous.fit)
    """

    def __init__(self, distribution: rv_continuous) -> None:
        """
        Init continuous univariate distribution with scipy.stats distribution.

        :param rv_continuous distribution: continuous univariate distribution from scipy.stats
        """

        self.distribution = distribution

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.distribution_params = self.distribution.fit(X)
        else:
            n_classes = max(y) + 1
            self.distribution_params = n_classes * [0]

            for cls in range(n_classes):
                self.distribution_params[cls] = self.distribution.fit(X[y == cls])

            self.distribution_params = np.array(self.distribution_params)
