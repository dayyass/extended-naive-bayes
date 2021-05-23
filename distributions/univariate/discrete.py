from typing import Optional

import numpy as np
from scipy import stats

from distributions.abstract import AbstractDistribution
from utils import to_categorical


class Bernoulli(AbstractDistribution):
    """
    Bernoulli distributions with parameter prob.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.prob = self.compute_prob_mle(X)
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls])  # type: ignore

    def predict_log_proba(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            log_proba = stats.bernoulli.logpmf(X, p=self.prob)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.prob)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.bernoulli.logpmf(X, p=self.prob[cls])  # type: ignore

        return log_proba

    @staticmethod
    def compute_prob_mle(X: np.ndarray) -> float:
        """
        Compute maximum likelihood estimator for parameter prob.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameter prob.
        :rtype: float
        """

        Bernoulli._check_univariate_input_data(X=X)

        prob = X.mean()
        return prob


class Categorical(AbstractDistribution):
    """
    Categorical distributions with parameters vector prob.
    """

    def __init__(self, k: int) -> None:
        """
        Init distribution with K possible categories.

        :param int k: number of possible categories.
        """

        assert k > 2, "for k = 2 use Bernoulli distribution."

        self.k = k

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.prob = self.compute_prob_mle(X, k=self.k)
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros((n_classes, self.k))

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls], k=self.k)  # type: ignore

    def predict_log_proba(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            log_proba = stats.multinomial.logpmf(
                to_categorical(X, num_classes=self.k), n=1, p=self.prob
            )
        else:
            n_samples = X.shape[0]
            n_classes = len(self.prob)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.multinomial.logpmf(
                    to_categorical(X, num_classes=self.k), n=1, p=self.prob[cls]
                )  # type: ignore

        return log_proba

    @staticmethod
    def compute_prob_mle(X: np.ndarray, k: int) -> np.ndarray:
        """
        Compute maximum likelihood estimator for parameters vector prob.

        :param np.ndarray X: training data.
        :param int k: number of possible categories.
        :return: maximum likelihood estimator for parameters vector prob.
        :rtype: np.ndarray
        """

        assert k > 2, "for k = 2 use Bernoulli distribution."
        Categorical._check_univariate_input_data(X=X)

        prob = np.zeros(k)
        for x in X:
            prob[x] += 1
        prob /= prob.sum()

        return prob


class Binomial(AbstractDistribution):
    """
    Binomial distributions with parameter prob.
    """

    def __init__(self, n: int) -> None:
        """
        Init distribution with N independent experiments.

        :param int n: number of independent experiments.
        """

        assert n > 1, "for n = 1 use Bernoulli distribution."

        self.n = n

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.prob = self.compute_prob_mle(X, n=self.n)
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls], n=self.n)  # type: ignore

    def predict_log_proba(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            log_proba = stats.binom.logpmf(X, n=self.n, p=self.prob)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.prob)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.binom.logpmf(X, n=self.n, p=self.prob[cls])  # type: ignore

        return log_proba

    @staticmethod
    def compute_prob_mle(X: np.ndarray, n: int) -> float:
        """
        Compute maximum likelihood estimator for parameter prob.

        :param np.ndarray X: training data.
        :param int n: number of independent experiments.
        :return: maximum likelihood estimator for parameter prob.
        :rtype: float
        """

        assert n > 1, "for n = 1 use Bernoulli distribution."
        Categorical._check_univariate_input_data(X=X)

        prob = X.mean() / n
        return prob


class Geometric(AbstractDistribution):
    """
    Geometric distributions with parameter prob.
    Probability distribution of the number X of Bernoulli trials needed to get one success.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.prob = self.compute_prob_mle(X)
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls])  # type: ignore

    def predict_log_proba(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            log_proba = stats.geom.logpmf(X, p=self.prob)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.prob)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.geom.logpmf(X, p=self.prob[cls])  # type: ignore

        return log_proba

    @staticmethod
    def compute_prob_mle(X: np.ndarray) -> float:
        """
        Compute maximum likelihood estimator for parameter prob.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameter prob.
        :rtype: float
        """

        Geometric._check_univariate_input_data(X=X)

        prob = 1 / X.mean()
        return prob


class Poisson(AbstractDistribution):
    """
    Poisson distributions with parameter lambda.
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

    def predict_log_proba(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            log_proba = stats.poisson.logpmf(X, mu=self.lambda_)
        else:
            n_samples = X.shape[0]
            n_classes = len(self.lambda_)  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.poisson.logpmf(X, mu=self.lambda_[cls])  # type: ignore

        return log_proba

    @staticmethod
    def compute_lambda_mle(X: np.ndarray) -> float:
        """
        Compute maximum likelihood estimator for parameter lambda.

        :param np.ndarray X: training data.
        :return: maximum likelihood estimator for parameter lambda.
        :rtype: float
        """

        Poisson._check_univariate_input_data(X=X)

        lambda_ = X.mean()
        return lambda_
