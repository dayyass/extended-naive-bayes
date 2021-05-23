from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# TODO: add method "predict_log_proba"
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
