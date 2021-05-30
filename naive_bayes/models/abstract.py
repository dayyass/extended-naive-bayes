from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from naive_bayes.distributions.abstract import AbstractDistribution
from naive_bayes.utils import isinteger


# TODO: compare with sklearn
class AbstractModel(ABC):
    """
    Abstract base class to represent Naive Bayes model.
    """

    def __init__(self, distributions: List[Union[AbstractDistribution, str]]) -> None:
        """
        Init model with distribution for each feature.
        Available distributions for ExtendedNaiveBayes in naive_bayes.distributions.
        Available distributions for SklearnExtendedNaiveBayes:
           ‘gaussian’
              normal distributed feature
           ‘bernoulli’
              bernoulli distributed feature
           ‘categorical’
              categorical distributed feature
           ‘multinomial’
              multinomial distributed feature

        :param List[Union[AbstractDistribution, str]] distributions: list of feature distributions.
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute model predictions.

        :param np.ndarray X: training data.
        :return: model predictions.
        :rtype: np.ndarray
        """

        log_prob_y_x = self.predict_log_proba(X)
        return np.argmax(log_prob_y_x, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute class probabilities.

        :param np.ndarray X: training data.
        :return: class probabilities.
        :rtype: np.ndarray
        """

        return np.exp(self.predict_log_proba(X))

    @abstractmethod
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute class log probabilities.

        :param np.ndarray X: training data.
        :return: class log probabilities.
        :rtype: np.ndarray
        """
        pass

    # TODO: raise exception if model is not fitted
    # TODO: add _check
    @abstractmethod
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate samples from fitted data.

        :param int n_samples: number of samples.
        :param Optional[int] random_state: random number generator seed.
        :return: samples.
        :rtype: np.ndarray
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Method to compute mean accuracy given X data and y labels.

        :param np.ndarray X: training data.
        :param np.ndarray y: target values.
        :return: mean accuracy.
        :rtype: float
        """

        self._check_input_data(X=X, y=y)

        return np.mean(self.predict(X) == y)

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
            assert min(y) == 0, "y labels should starts with 0."
            assert isinteger(y), "y should be integer vector."
