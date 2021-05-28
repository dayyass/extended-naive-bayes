from typing import Optional

import numpy as np
from scipy.stats import rv_continuous
from sklearn.neighbors import KernelDensity

from naive_bayes.distributions.abstract import AbstractDistribution
from naive_bayes.distributions.univariate.continuous import (  # noqa: F401
    Beta,
    Exponential,
    Gamma,
    Normal,
)
from naive_bayes.distributions.univariate.discrete import (  # noqa: F401
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
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X)

        if y is None:
            self.distribution_params = self.distribution.fit(X)
        else:
            n_classes = max(y) + 1
            self.distribution_params = n_classes * [0]

            for cls in range(n_classes):
                self.distribution_params[cls] = self.distribution.fit(X[y == cls])

            self.distribution_params = np.array(self.distribution_params)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X)

        if not isinstance(self.distribution_params, np.ndarray):
            log_proba = self.distribution.logpdf(X, *self.distribution_params)
        else:
            n_samples = X.shape[0]
            n_classes = self.distribution_params.shape[0]  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = self.distribution.logpdf(X, *self.distribution_params[cls])  # type: ignore

        return log_proba

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """

        if not isinstance(self.distribution_params, np.ndarray):
            samples = self.distribution.rvs(
                *self.distribution_params, size=n_samples, random_state=random_state
            )
        else:
            n_classes = self.distribution_params.shape[0]  # type: ignore
            samples = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                samples[:, cls] = self.distribution.rvs(*self.distribution_params[cls], size=n_samples, random_state=random_state)  # type: ignore

        return samples

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        pass


# TODO: add to multivariate
class KernelDensityEstimator(AbstractDistribution):
    """
    Kernel Density Estimation (Parzen–Rosenblatt window method) - non-parametric method.
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        metric: str = "euclidean",
    ) -> None:
        """
        Init Kernel Density Model.

        :param float bandwidth: The bandwidth of the kernel.
        :param str kernel: The kernel to use.
        :param str metric: The distance metric to use.
        """

        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the Kernel Density model on the data.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X)

        if y is None:
            self.kde = KernelDensity(
                bandwidth=self.bandwidth, kernel=self.kernel, metric=self.metric
            ).fit(X[:, np.newaxis])
        else:
            n_classes = max(y) + 1
            self.kde = n_classes * [0]

            for cls in range(n_classes):
                self.kde[cls] = KernelDensity(
                    bandwidth=self.bandwidth, kernel=self.kernel, metric=self.metric
                ).fit(X[y == cls][:, np.newaxis])

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the log density model on the data.

        :param np.ndarray X: data.
        :return: log density on the data.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X)

        if not isinstance(self.kde, list):
            log_proba = self.kde.score_samples(X[:, np.newaxis])
        else:
            n_samples = X.shape[0]
            n_classes = len(self.kde)
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = self.kde[cls].score_samples(X[:, np.newaxis])

        return log_proba

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """

        if not isinstance(self.kde, list):
            samples = self.kde.sample(n_samples=n_samples, random_state=random_state)
        else:
            n_classes = len(self.kde)
            samples = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                samples[:, cls] = self.kde[cls].sample(n_samples=n_samples, random_state=random_state)  # type: ignore

        return samples

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        pass
