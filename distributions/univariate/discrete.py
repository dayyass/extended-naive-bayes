from typing import Optional

import numpy as np

from distributions.abstract import AbstractDistribution


class Bernoulli(AbstractDistribution):
    """
    Bernoulli distributions with parameter prob.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        self._check_univariate_input_data(X=X, y=y)

        if y is None:
            self.prob = X.mean()
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = X[y == cls].mean()


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
            self.prob = np.zeros(self.k)
            for x in X:
                self.prob[x] += 1
            self.prob /= self.prob.sum()
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros((n_classes, self.k))

            for cls in range(n_classes):
                for x in X[y == cls]:
                    self.prob[cls][x] += 1
            self.prob /= self.prob.sum(axis=1)[:, np.newaxis]


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

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        if y is None:
            self.prob = X.mean() / self.n
        else:
            n_classes = max(y) + 1
            self.prob = np.zeros(n_classes)

            for cls in range(n_classes):
                self.prob[cls] = X[y == cls].mean() / self.n
