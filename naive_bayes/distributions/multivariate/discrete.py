from typing import Optional

import numpy as np
from scipy import stats

from naive_bayes.distributions.abstract import AbstractDistribution
from naive_bayes.utils import isinteger


class Multinomial(AbstractDistribution):
    """
    Multinomial distribution with parameters n and vector prob.
    """

    def __init__(self, n: int) -> None:
        """
        Init distribution with N independent experiments.

        :param int n: number of independent experiments.
        """

        assert n > 1, "for n = 1 use Categorical distribution."

        self.n = n

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        self._check_input_data(X=X, y=y, univariate=False)
        self._check_support(X=X, n=self.n)

        if y is None:
            self.prob = self.compute_prob_mle(X, n=self.n)
        else:
            n_classes = max(y) + 1
            n_features = X.shape[1]
            self.prob = np.zeros((n_classes, n_features))

            for cls in range(n_classes):
                self.prob[cls] = self.compute_prob_mle(X[y == cls], n=self.n)  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X, univariate=False)
        self._check_support(X=X, n=self.n)

        if self.prob.ndim == 1:
            log_proba = stats.multinomial.logpmf(X, n=self.n, p=self.prob)
        else:
            n_samples = X.shape[0]
            n_classes = self.prob.shape[0]  # type: ignore
            log_proba = np.zeros((n_samples, n_classes))

            for cls in range(n_classes):
                log_proba[:, cls] = stats.multinomial.logpmf(X, n=self.n, p=self.prob[cls])  # type: ignore

        return log_proba

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """

        if self.prob.ndim == 1:
            samples = stats.multinomial.rvs(
                n=self.n, p=self.prob, size=n_samples, random_state=random_state
            )
        else:
            n_classes, n_features = self.prob.shape
            samples = np.zeros((n_samples, n_features, n_classes))

            for cls in range(n_classes):
                samples[:, :, cls] = stats.multinomial.rvs(n=self.n, p=self.prob[cls], size=n_samples, random_state=random_state)  # type: ignore

        return samples

    @staticmethod
    def compute_prob_mle(X: np.ndarray, n: int) -> np.ndarray:
        """
        Compute maximum likelihood estimator for parameters vector prob.

        :param np.ndarray X: training data.
        :param int n: number of independent experiments.
        :return: maximum likelihood estimator for parameter prob.
        :rtype: np.ndarray
        """

        assert n > 1, "for n = 1 use Categorical distribution."
        Multinomial._check_input_data(X=X, univariate=False)
        Multinomial._check_support(X=X, n=n)

        prob = X.mean(axis=0) / n
        return prob

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
        assert np.all(
            X.sum(axis=1) == kwargs["n"]
        ), f"sum of each row should be equal to {kwargs['n']}."
