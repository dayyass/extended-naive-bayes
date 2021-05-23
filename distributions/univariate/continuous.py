from typing import Optional

import numpy as np

from distributions.abstract import AbstractDistribution


class Gaussian(AbstractDistribution):
    """
    Gaussian (Normal) distributions with parameters mu and sigma.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

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


class Exponential(AbstractDistribution):
    """
    Exponential distributions with parameter lambda.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        if y is None:
            self.lambda_ = 1 / X.mean()
        else:
            n_classes = max(y) + 1
            self.lambda_ = np.zeros(n_classes)

            for cls in range(n_classes):
                self.lambda_[cls] = 1 / X[y == cls].mean()


class Gamma(AbstractDistribution):
    """
    Gamma distributions with parameters alpha and beta.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        if y is None:
            self.alpha = self._compute_alpha_mme(X)
            self.beta = self._compute_beta_mme(X)
        else:
            n_classes = max(y) + 1
            self.alpha = np.zeros(n_classes)
            self.beta = np.zeros(n_classes)

            for cls in range(n_classes):
                self.alpha[cls] = self._compute_alpha_mme(X[y == cls])  # type: ignore
                self.beta[cls] = self._compute_beta_mme(X[y == cls])  # type: ignore

    @staticmethod
    def _compute_alpha_mme(X: np.ndarray) -> float:
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
    def _compute_beta_mme(X: np.ndarray) -> float:
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
