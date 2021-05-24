from typing import List

import numpy as np
from scipy.special import logsumexp

from distributions import Bernoulli, Categorical, Normal
from distributions.abstract import AbstractDistribution
from models.abstract import AbstractModel


class NaiveBayes(AbstractModel):
    """
    Naive Bayes model.
    """

    def __init__(self, distributions: List[AbstractDistribution]) -> None:
        """
        Init model with distributions for all features.

        :param List[AbstractDistribution] distributions: list of feature distributions.
        """

        super().__init__(distributions)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Method to fit the model.

        :param np.ndarray X: training data.
        :param np.ndarray y: target values.
        """

        self._check_input_data(X=X, y=y)

        # priors
        _, counts = np.unique(y, return_counts=True)
        self.priors = counts / counts.sum()

        # distributions
        for feature in range(len(self.distributions)):
            self.distributions[feature].fit(X[:, feature], y)

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

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute class log probabilities.

        :param np.ndarray X: training data.
        :return: class log probabilities.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)

        n_classes = len(self.priors)
        n_samples = X.shape[0]

        log_prob_y = np.log(self.priors)
        log_prob_xy = np.zeros((n_samples, n_classes))

        for feature in range(len(self.distributions)):
            log_prob_xy += self.distributions[feature].predict_log_proba(X[:, feature])
        log_prob_xy += log_prob_y  # TODO: start init (np.repeat)

        log_prob_x = logsumexp(log_prob_xy, axis=1)
        log_prob_y_x = log_prob_xy - log_prob_x[:, np.newaxis]

        return log_prob_y_x

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        :param np.ndarray X: training data.
        :param np.ndarray y: target values.
        :return: mean accuracy.
        :rtype: float
        """

        self._check_input_data(X=X, y=y)

        return np.mean(self.predict(X) == y)


class GaussianNaiveBayes(NaiveBayes):
    """
    Naive Bayes model with normal distributed features.
    """

    def __init__(self, n_features: int) -> None:
        """
        Init model with {n_features} normal distributed features.

        :param int n_features: number of features.
        """

        super().__init__(distributions=[Normal() for _ in range(n_features)])


class BernoulliNaiveBayes(NaiveBayes):
    """
    Naive Bayes model with bernoulli distributed features.
    """

    def __init__(self, n_features: int) -> None:
        """
        Init model with {n_features} bernoulli distributed features.

        :param int n_features: number of features.
        """

        super().__init__(distributions=[Bernoulli() for _ in range(n_features)])


class CategoricalNaiveBayes(NaiveBayes):
    """
    Naive Bayes model with categorical distributed features.
    """

    def __init__(self, n_features: int, n_categories: List[int]) -> None:
        """
        Init model with {n_features} categorical distributed features.

        :param int n_features: number of features.
        :param List[int] n_categories: number of categories for each feature.
        """

        assert (
            len(n_categories) == n_features
        ), "length of n_categories should be equal n_features."

        super().__init__(
            distributions=[Categorical(n_categories[i]) for i in range(n_features)]
        )
