from typing import Optional

import numpy as np
from scipy.stats import rv_continuous
from sklearn.mixture import GaussianMixture
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
    Based on sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        metric: str = "euclidean",
        **kwargs,
    ) -> None:
        """
        Init Kernel Density Model.

        :param float bandwidth: The bandwidth of the kernel.
        :param str kernel: The kernel to use.
        :param str metric: The distance metric to use.
        :param kwargs: additional sklearn Kernel Density Model parameters.
        """

        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.kwargs = kwargs

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
                bandwidth=self.bandwidth,
                kernel=self.kernel,
                metric=self.metric,
                **self.kwargs,
            ).fit(X[:, np.newaxis])
        else:
            n_classes = max(y) + 1
            self.kde = n_classes * [0]

            for cls in range(n_classes):
                self.kde[cls] = KernelDensity(
                    bandwidth=self.bandwidth,
                    kernel=self.kernel,
                    metric=self.metric,
                    **self.kwargs,
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
            samples = self.kde.sample(
                n_samples=n_samples, random_state=random_state
            ).squeeze()
        else:
            n_classes = len(self.kde)
            samples = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                samples[:, cls] = self.kde[cls].sample(n_samples=n_samples, random_state=random_state).squeeze()  # type: ignore

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
class GaussianMixtureEstimator(AbstractDistribution):
    """
    Gaussian Mixture Estimation.
    Based on sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Init Gaussian Mixture Model.

        :param int n_components: The number of mixture components.
        :param str covariance_type: String describing the type of covariance parameters to use. Must be one of:
           ‘full’
              each component has its own variance
           ‘tied’
              all components share the same variance
           Note: sklearn GaussianMixture allows 4 types of covariance_type: ‘full’, ‘tied’, ‘diag’, ‘spherical’,
                 but in univariate case, ‘full’, ‘diag’ and ‘spherical’ lead to the same result,
                 so they collapsed into ‘full’ covariance_type.
        :param Optional[int] random_state: random number generator seed.
        :param kwargs: additional sklearn Gaussian Mixture Model parameters.
        """

        assert n_components > 1, "for n_components = 1 use Normal distribution."
        assert covariance_type in [
            "full",
            "tied",
        ], "covariance type should by ‘full’ or ‘tied’"

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the Gaussian Mixture model on the data.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X)

        if y is None:
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                **self.kwargs,
            ).fit(X[:, np.newaxis])
        else:
            n_classes = max(y) + 1
            self.gmm = n_classes * [0]

            for cls in range(n_classes):
                self.gmm[cls] = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    random_state=self.random_state,
                    **self.kwargs,
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

        if not isinstance(self.gmm, list):
            log_proba = self.gmm.score_samples(X[:, np.newaxis])
        else:
            n_samples = X.shape[0]
            n_classes = len(self.gmm)
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = self.gmm[cls].score_samples(X[:, np.newaxis])

        return log_proba

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """

        if not isinstance(self.gmm, list):
            samples = self.gmm.sample(n_samples=n_samples)[0].squeeze()
        else:
            n_classes = len(self.gmm)
            samples = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                samples[:, cls] = self.gmm[cls].sample(n_samples=n_samples)[0].squeeze()  # type: ignore

        return samples

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        pass
