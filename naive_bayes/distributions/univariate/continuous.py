from typing import Optional

import numpy as np
from scipy import stats

from naive_bayes.distributions.abstract import AbstractDistribution


class Normal(AbstractDistribution):
    """
    Normal (Gaussian) distribution with parameters mu and sigma.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X)

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
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X)

        if not isinstance(self.mu, np.ndarray):
            log_proba = stats.norm.logpdf(X, loc=self.mu, scale=self.sigma)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.mu)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.norm.logpdf(X, loc=self.mu[cls], scale=self.sigma[cls])  # type: ignore

        return log_proba

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """

        if not isinstance(self.mu, np.ndarray):
            samples = stats.norm.rvs(
                loc=self.mu, scale=self.sigma, size=n_samples, random_state=random_state
            )
        else:
            n_classes = len(self.mu)  # type: ignore
            samples = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                samples[:, cls] = stats.norm.rvs(loc=self.mu[cls], scale=self.sigma[cls], size=n_samples, random_state=random_state)  # type: ignore

        return samples

    @staticmethod
    def compute_mu_mle(X: np.ndarray) -> float:
        """
        Compute maximum likelihood estimator for parameter mu.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameter mu.
        :rtype: float
        """

        Normal._check_input_data(X=X)
        Normal._check_support(X=X)

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

        Normal._check_input_data(X=X)
        Normal._check_support(X=X)

        sigma = X.std()
        return sigma

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        pass


class Exponential(AbstractDistribution):
    """
    Exponential distribution with parameter lambda.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X)

        if y is None:
            self.lambda_ = self.compute_lambda_mle(X)
        else:
            n_classes = max(y) + 1
            self.lambda_ = np.zeros(n_classes)

            for cls in range(n_classes):
                self.lambda_[cls] = self.compute_lambda_mle(X[y == cls])  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X)

        if not isinstance(self.lambda_, np.ndarray):
            log_proba = stats.expon.logpdf(X, scale=1 / self.lambda_)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.lambda_)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.expon.logpdf(X, scale=1 / self.lambda_[cls])  # type: ignore

        return log_proba

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """

        if not isinstance(self.lambda_, np.ndarray):
            samples = stats.expon.rvs(
                scale=1 / self.lambda_, size=n_samples, random_state=random_state
            )
        else:
            n_classes = len(self.lambda_)  # type: ignore
            samples = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                samples[:, cls] = stats.expon.rvs(scale=1 / self.lambda_[cls], size=n_samples, random_state=random_state)  # type: ignore

        return samples

    @staticmethod
    def compute_lambda_mle(X: np.ndarray) -> float:
        """
        Compute maximum likelihood estimator for parameter lambda.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameter lambda.
        :rtype: float
        """

        Exponential._check_input_data(X=X)
        Exponential._check_support(X=X)

        lambda_ = 1 / X.mean()
        return lambda_

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        assert np.all(X >= 0), "x should be greater or equal to 0."


class Gamma(AbstractDistribution):
    """
    Gamma distribution with parameters alpha and beta.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X)

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
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X)

        if not isinstance(self.alpha, np.ndarray):
            log_proba = stats.gamma.logpdf(X, a=self.alpha, scale=1 / self.beta)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.alpha)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.gamma.logpdf(X, a=self.alpha[cls], scale=1 / self.beta[cls])  # type: ignore

        return log_proba

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """

        if not isinstance(self.alpha, np.ndarray):
            samples = stats.gamma.rvs(
                a=self.alpha,
                scale=1 / self.beta,
                size=n_samples,
                random_state=random_state,
            )
        else:
            n_classes = len(self.alpha)  # type: ignore
            samples = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                samples[:, cls] = stats.gamma.rvs(a=self.alpha[cls], scale=1 / self.beta[cls], size=n_samples, random_state=random_state)  # type: ignore

        return samples

    @staticmethod
    def compute_alpha_mme(X: np.ndarray) -> float:
        """
        Compute mixed type log-moment estimator for parameter alpha.

        :param np.ndarray X: training data.
        :return: mixed type log-moment estimator for parameter alpha.
        :rtype: float
        """

        Gamma._check_input_data(X=X)
        Gamma._check_support(X=X)

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

        Gamma._check_input_data(X=X)
        Gamma._check_support(X=X)

        n = X.shape[0]

        theta = (n * (X * np.log(X)).sum() - X.sum() * np.log(X).sum()) / n ** 2
        theta *= n / (n - 1)
        beta = 1 / theta

        return beta

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        assert np.all(X >= 0), "x should be greater or equal to 0."


class Beta(AbstractDistribution):
    """
    Beta distribution with parameters alpha and beta.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X)

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
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X)

        if not isinstance(self.alpha, np.ndarray):
            log_proba = stats.beta.logpdf(X, a=self.alpha, b=self.beta)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.alpha)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.beta.logpdf(X, a=self.alpha[cls], b=self.beta[cls])  # type: ignore

        return log_proba

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """

        if not isinstance(self.alpha, np.ndarray):
            samples = stats.beta.rvs(
                a=self.alpha, b=self.beta, size=n_samples, random_state=random_state
            )
        else:
            n_classes = len(self.alpha)  # type: ignore
            samples = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                samples[:, cls] = stats.beta.rvs(a=self.alpha[cls], b=self.beta[cls], size=n_samples, random_state=random_state)  # type: ignore

        return samples

    @staticmethod
    def compute_alpha_mme(X: np.ndarray) -> float:
        """
        Compute method of moments estimator for parameter alpha.

        :param np.ndarray X: training data.
        :return: method of moments estimator for parameter alpha.
        :rtype: float
        """

        Beta._check_input_data(X=X)
        Beta._check_support(X=X)

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

        Beta._check_input_data(X=X)
        Beta._check_support(X=X)

        mean = X.mean()
        var = X.var(ddof=1)  # unbiased estimator of the variance

        # TODO: fix with mle
        assert var < mean * (
            1 - mean
        ), "method of moments estimator conditions violated"

        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        return beta

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        assert np.all(
            (X >= 0) & (X <= 1)
        ), "x should be between 0 and 1 (both inclusive)."
