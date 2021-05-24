from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


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

        :param np.ndarray X: training data.
        :return: log probabilities for X.
        :rtype: np.ndarray
        """
        pass

    @staticmethod
    def _check_input_data(
        X: np.ndarray, y: Optional[np.ndarray] = None, univariate: bool = True
    ) -> None:
        """
        Method to check correctness of input data to fit method for univariate distributions.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        if univariate:
            assert X.ndim == 1, "X should be a 1d vector."
        else:
            assert X.ndim == 2, "X should be a 2d matrix."

        if y is not None:
            assert y.ndim == 1, "y should be a 1d vector."
