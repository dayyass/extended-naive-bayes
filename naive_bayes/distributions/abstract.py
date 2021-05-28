from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from naive_bayes.utils import isinteger


# TODO: hide from user
class AbstractDistribution(ABC):
    """
    Abstract base class to represent probability distributions.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to compute MLE given X (data). If y is provided, computes MLE of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """
        pass

    @abstractmethod
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).

        :param np.ndarray X: data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """
        pass

    # TODO: raise exception if distribution is not fitted
    # TODO: add _check
    @abstractmethod
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variables samples from fitted distribution.

        :param int n_samples: number of random variables samples.
        :param Optional[int] random_state: random number generator seed.
        :return: random variables samples.
        :rtype: np.ndarray
        """
        pass

    @staticmethod
    def _check_input_data(
        X: np.ndarray, y: Optional[np.ndarray] = None, univariate: bool = True
    ) -> None:
        """
        Method to check correctness of input data X and y.

        :param np.ndarray X: data.
        :param Optional[np.ndarray] y: target values.
        """

        if univariate:
            assert X.ndim == 1, "X should be a 1d vector."
        else:
            assert X.ndim == 2, "X should be a 2d matrix."

        if y is not None:
            assert y.ndim == 1, "y should be a 1d vector."
            assert min(y) == 0, "y labels should starts with 0."
            assert isinteger(y), "y should be integer vector."

    @staticmethod
    @abstractmethod
    def _check_support(X: np.ndarray, **kwargs) -> None:
        """
        Method to check data for being in random variable support.

        :param np.ndarray X: data.
        :param kwargs: additional distribution parameters.
        """
        pass
