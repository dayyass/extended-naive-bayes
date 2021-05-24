from typing import Optional

import numpy as np
from scipy import stats

from distributions.abstract import AbstractDistribution
from utils import isinteger, to_categorical


class Bernoulli(AbstractDistribution):
    """
    Bernoulli distribution with parameter prob.
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
            self.prob = self.compute_prob_mle(X)
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls])  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X)

        if not isinstance(self.prob, np.ndarray):
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

        Bernoulli._check_input_data(X=X)
        Bernoulli._check_support(X=X)

        prob = X.mean()
        return prob

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        assert ((X == 0) | (X == 1)).all(), "x should be equal to 0 or 1."


class Categorical(AbstractDistribution):
    """
    Categorical distribution with parameters vector prob.
    """

    def __init__(self, k: int) -> None:
        """
        Init distribution with K possible categories.

        :param int k: number of possible categories.
        """

        assert k > 2, "for k = 2 use Bernoulli distribution."

        self.k = k

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X, k=self.k)

        if y is None:
            self.prob = self.compute_prob_mle(X, k=self.k)
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros((n_classes, self.k))

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls], k=self.k)  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X, k=self.k)

        if self.prob.ndim == 1:
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
        Categorical._check_input_data(X=X)
        Categorical._check_support(X=X, k=k)

        prob = np.zeros(k)
        for x in X:
            prob[x] += 1
        prob /= prob.sum()

        return prob

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        X_union = (X == 0) | (X == 1)
        for k in range(2, kwargs["k"]):
            X_union = X_union | (X == k)

        assert (
            X_union.all()
        ), f"x should be equal to integer from 0 to {kwargs['k']} (exclusive)."


class Binomial(AbstractDistribution):
    """
    Binomial distribution with parameter prob.
    """

    def __init__(self, n: int) -> None:
        """
        Init distribution with N independent experiments.

        :param int n: number of independent experiments.
        """

        assert n > 1, "for n = 1 use Bernoulli distribution."

        self.n = n

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y)
        self._check_support(X=X, n=self.n)

        if y is None:
            self.prob = self.compute_prob_mle(X, n=self.n)
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls], n=self.n)  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X, n=self.n)

        if not isinstance(self.prob, np.ndarray):
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
        Binomial._check_input_data(X=X)
        Binomial._check_support(X=X, n=n)

        prob = X.mean() / n
        return prob

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        X_union = (X == 0) | (X == 1)
        for k in range(2, kwargs["n"] + 1):
            X_union = X_union | (X == k)

        assert (
            X_union.all()
        ), f"x should be equal to integer from 0 to {kwargs['n']} (inclusive)."


class Geometric(AbstractDistribution):
    """
    Geometric distribution with parameter prob.
    Probability distribution of the number X of Bernoulli trials needed to get one success.
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
            self.prob = self.compute_prob_mle(X)
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls])  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)
        self._check_support(X=X)

        if not isinstance(self.prob, np.ndarray):
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

        Geometric._check_input_data(X=X)
        Geometric._check_support(X=X)

        prob = 1 / X.mean()
        return prob

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        assert (X > 0).all() & isinteger(X), "x should be greater then 0 and integer."


class Poisson(AbstractDistribution):
    """
    Poisson distribution with parameter lambda.
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

        Poisson._check_input_data(X=X)
        Poisson._check_support(X=X)

        lambda_ = X.mean()
        return lambda_

    @staticmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """

        assert (X >= 0).all() & isinteger(
            X
        ), "x should be greater or equal to 0 and integer."
