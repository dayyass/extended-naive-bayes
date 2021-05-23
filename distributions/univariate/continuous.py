from typing import Optional

import numpy as np

from distributions.abstract import AbstractDistribution


class Gaussian(AbstractDistribution):
    """
    Gaussian (Normal) distributions with parameters mu and sigma.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.mu = self.compute_mu_mle(X)
            self.sigma = self.compute_sigma_mle(X)
        else:
            n_classes = max(y) + 1
            self.mu = np.zeros(n_classes)
            self.sigma = np.zeros(n_classes)

            for cls in range(n_classes):
                self.mu[cls] = self.compute_mu_mle(X[y == cls])  # type: ignore
                self.sigma[cls] = self.compute_sigma_mle(X[y == cls])  # type: ignore

    @staticmethod
    def compute_mu_mle(X: np.ndarray) -> float:
        """
        Compute maximum likelihood estimator for parameter mu.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameter mu.
        :rtype: float
        """

        mu = X.mean()
        return mu

    @staticmethod
    def compute_sigma_mle(X: np.ndarray) -> float:
        """
        Compute maximum likelihood estimator for parameter sigma.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameter sigma.
        :rtype: float
        """

        sigma = X.std()
        return sigma


class Exponential(AbstractDistribution):
    """
    Exponential distributions with parameter lambda.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.lambda_ = self.compute_lambda_mle(X)
        else:
            n_classes = max(y) + 1
            self.lambda_ = np.zeros(n_classes)

            for cls in range(n_classes):
                self.lambda_[cls] = self.compute_lambda_mle(X[y == cls])  # type: ignore

    @staticmethod
    def compute_lambda_mle(X: np.ndarray) -> float:
        """
        Compute maximum likelihood estimator for parameter lambda.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameter lambda.
        :rtype: float
        """

        lambda_ = 1 / X.mean()
        return lambda_


class Gamma(AbstractDistribution):
    """
    Gamma distributions with parameters alpha and beta.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.alpha = self.compute_alpha_mme(X)
            self.beta = self.compute_beta_mme(X)
        else:
            n_classes = max(y) + 1
            self.alpha = np.zeros(n_classes)
            self.beta = np.zeros(n_classes)

            for cls in range(n_classes):
                self.alpha[cls] = self.compute_alpha_mme(X[y == cls])  # type: ignore
                self.beta[cls] = self.compute_beta_mme(X[y == cls])  # type: ignore

    @staticmethod
    def compute_alpha_mme(X: np.ndarray) -> float:
        """
        Compute mixed type log-moment estimator for parameter alpha.

        :param np.ndarray X: training data.
        :return: mixed type log-moment estimator for parameter alpha.
        :rtype: float
        """

        n = X.shape[0]

        alpha = n * X.sum() / (n * (X * np.log(X)).sum() - X.sum() * np.log(X).sum())
        alpha -= (
            3 * alpha - 2 / 3 * alpha / (1 + alpha) - 4 / 5 * alpha / (1 + alpha) ** 2
        ) / n

        return alpha

    @staticmethod
    def compute_beta_mme(X: np.ndarray) -> float:
        """
        Compute mixed type log-moment estimator for parameter beta.

        :param np.ndarray X: training data.
        :return: mixed type log-moment estimator for parameter beta.
        :rtype: float
        """

        n = X.shape[0]

        theta = (n * (X * np.log(X)).sum() - X.sum() * np.log(X).sum()) / n ** 2
        theta *= n / (n - 1)
        beta = 1 / theta

        return beta
