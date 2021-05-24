from typing import Optional

import numpy as np
from scipy import stats

from distributions.abstract import AbstractDistribution


class MultivariateNormal(AbstractDistribution):
    """
    Multivariate Normal (gaussian) distribution with parameters mu and sigma.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y, univariate=False)
        self._check_support(X=X)

        if y is None:
            self.mu = self.compute_mu_mle(X)
            self.sigma = self.compute_sigma_mle(X)
        else:
            n_classes = max(y) + 1
            self.mu = np.zeros((n_classes, X.shape[1]))
            self.sigma = np.zeros((n_classes, X.shape[1], X.shape[1]))

            for cls in range(n_classes):
                self.mu[cls] = self.compute_mu_mle(X[y == cls])  # type: ignore
                self.sigma[cls] = self.compute_sigma_mle(X[y == cls])  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X, univariate=False)
        self._check_support(X=X)

        if self.mu.ndim == 1:
            log_proba = stats.multivariate_normal.logpdf(
                X, mean=self.mu, cov=self.sigma
            )
        else:
            n_samples = X.shape[0]
            n_classes = self.mu.shape[0]  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.multivariate_normal.logpdf(X, mean=self.mu[cls], cov=self.sigma[cls])  # type: ignore

        return log_proba

    @staticmethod
    def compute_mu_mle(X: np.ndarray) -> np.ndarray:
        """
        Compute maximum likelihood estimator for parameters vector mu.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameters vector mu.
        :rtype: np.ndarray
        """

        MultivariateNormal._check_input_data(X=X, univariate=False)
        MultivariateNormal._check_support(X=X)

        mu = X.mean(axis=0)
        return mu

    @staticmethod
    def compute_sigma_mle(X: np.ndarray) -> np.ndarray:
        """
        Compute maximum likelihood estimator for parameters matrix sigma.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameters matrix sigma.
        :rtype: np.ndarray
        """

        MultivariateNormal._check_input_data(X=X, univariate=False)
        MultivariateNormal._check_support(X=X)

        mu = X.mean(axis=0)
        sigma = (X - mu).T @ (X - mu) / X.shape[0]
        return sigma

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        pass
