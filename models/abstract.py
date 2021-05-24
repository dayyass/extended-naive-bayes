from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from distributions.abstract import AbstractDistribution


# TODO: compare with sklearn
class AbstractModel(ABC):
    """
    Abstract base class to represent Naive Bayes model.
    """

    def __init__(self, distributions: List[AbstractDistribution]) -> None:
        """
        Init model with distributions for all features.

        :param List[AbstractDistribution] distributions: list of feature distributions.
        """

        self.distributions = distributions

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Method to fit the model.

        :param np.ndarray X: training data.
        :param np.ndarray y: target values.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute model predictions.

        :param np.ndarray X: training data.
        :return: model predictions.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute class probabilities.

        :param np.ndarray X: training data.
        :return: class probabilities.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute class log probabilities.

        :param np.ndarray X: training data.
        :return: class log probabilities.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        :param np.ndarray X: training data.
        :param np.ndarray y: target values.
        :return: mean accuracy.
        :rtype: float
        """
        pass

    def _check_input_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Method to check correctness of input data.

        :param np.ndarray X: training data.
        :param Optional[np.ndarray] y: target values.
        """

        assert X.shape[1] == len(
            self.distributions
        ), "number of features should be equal to the number of distributions"
        if y is not None:
            assert y.ndim == 1, "y should be a 1d vector."
