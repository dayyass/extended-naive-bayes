from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# TODO: add method "predict_log_proba"
class AbstractDistribution(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        pass
