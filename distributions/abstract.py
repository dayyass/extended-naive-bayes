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
    def predict_log_proba(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Method to compute log probabilities given X (data).
        If y is provided, computes log probabilities of X for each class y.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """
        pass

    @staticmethod
    def _check_univariate_input_data(
        X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> None:
        """
        Method to check correctness of input data to fit method for univariate distributions.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        assert X.ndim == 1, "X should be a 1d vector."
        if y is not None:
            assert y.ndim == 1, "y should be a 1d vector."
