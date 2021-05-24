from typing import Optional

import numpy as np
from scipy import special, stats

from distributions.abstract import AbstractDistribution


class Gaussian(AbstractDistribution):
    """
    Gaussian (Normal) distributions with parameters mu and sigma.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_input_data(X=X, y=y)

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

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_input_data(X=X)

        if not isinstance(self.mu, np.ndarray):
            log_proba = stats.norm.logpdf(X, loc=self.mu, scale=self.sigma)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.mu)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.norm.logpdf(X, loc=self.mu[cls], scale=self.sigma[cls])  # type: ignore

        return log_proba

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

        self._check_input_data(X=X, y=y)

        if y is None:
            self.lambda_ = self.compute_lambda_mle(X)
        else:
            n_classes = max(y) + 1
            self.lambda_ = np.zeros(n_classes)

            for cls in range(n_classes):
                self.lambda_[cls] = self.compute_lambda_mle(X[y == cls])  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_input_data(X=X)

        if not isinstance(self.lambda_, np.ndarray):
            log_proba = self.logpdf(X, lambda_=self.lambda_)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.lambda_)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = self.logpdf(X, lambda_=self.lambda_[cls])  # type: ignore

        return log_proba

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

    @staticmethod
    def logpdf(X: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Compute log of the probability density function at X.

        :param np.ndarray  X: training data.
        :param float lambda_: parameter lambda.
        :return: log of the probability density function at X.
        :rtype: np.ndarray
        """

        logpdf = np.log(lambda_) - lambda_ * X
        return logpdf


class Gamma(AbstractDistribution):
    """
    Gamma distributions with parameters alpha and beta.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_input_data(X=X, y=y)

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

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_input_data(X=X)

        if not isinstance(self.alpha, np.ndarray):
            log_proba = self.logpdf(X, alpha=self.alpha, beta=self.beta)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.alpha)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = self.logpdf(X, alpha=self.alpha[cls], beta=self.beta[cls])  # type: ignore

        return log_proba

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

    @staticmethod
    def logpdf(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Compute log of the probability density function at X.

        :param np.ndarray  X: training data.
        :param float alpha: parameter alpha.
        :param float beta: parameter beta.
        :return: log of the probability density function at X.
        :rtype: np.ndarray
        """

        logpdf = (
            alpha * np.log(beta)
            - special.loggamma(alpha)
            + (alpha - 1) * np.log(X)
            - beta * X
        )
        return logpdf


class Beta(AbstractDistribution):
    """
    Beta distributions with parameters alpha and beta.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_input_data(X=X, y=y)

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

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:

        self._check_input_data(X=X)

        if not isinstance(self.alpha, np.ndarray):
            log_proba = stats.beta.logpdf(X, a=self.alpha, b=self.beta)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.alpha)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.beta.logpdf(X, a=self.alpha[cls], b=self.beta[cls])  # type: ignore

        return log_proba

    @staticmethod
    def compute_alpha_mme(X: np.ndarray) -> float:
        """
        Compute method of moments estimator for parameter alpha.

        :param np.ndarray X: training data.
        :return: method of moments estimator for parameter alpha.
        :rtype: float
        """

        mean = X.mean()
        var = X.var(ddof=1)  # unbiased estimator of the variance

        # TODO: fix with mle
        assert var < mean * (
            1 - mean
        ), "method of moments estimator conditions violated"

        alpha = mean * (mean * (1 - mean) / var - 1)
        return alpha

    @staticmethod
    def compute_beta_mme(X: np.ndarray) -> float:
        """
        Compute method of moments estimator for parameter beta.

        :param np.ndarray X: training data.
        :return: method of moments estimator for parameter beta.
        :rtype: float
        """

        mean = X.mean()
        var = X.var(ddof=1)  # unbiased estimator of the variance

        # TODO: fix with mle
        assert var < mean * (
            1 - mean
        ), "method of moments estimator conditions violated"

        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        return beta
