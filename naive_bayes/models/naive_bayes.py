from typing import List, Optional

import numpy as np
from scipy.special import logsumexp

from naive_bayes.distributions import Bernoulli, Categorical, Normal
from naive_bayes.models.abstract import AbstractModel


# TODO: add str parametrization for ExtendedNaiveBayes
class ExtendedNaiveBayes(AbstractModel):
    """
    Extended (allow different distributions for each feature) Naive Bayes model.
    """

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
            self.distributions[feature].fit(X[:, feature], y)  # type: ignore

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute class log probabilities.

        :param np.ndarray X: training data.
        :return: class log probabilities.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)

        n_samples = X.shape[0]

        log_prob_y = np.log(self.priors)
        log_prob_xy = np.repeat(log_prob_y[np.newaxis, :], repeats=n_samples, axis=0)

        for feature in range(len(self.distributions)):
            log_prob_xy += self.distributions[feature].predict_log_proba(X[:, feature])  # type: ignore

        log_prob_x = logsumexp(log_prob_xy, axis=1)
        log_prob_y_x = log_prob_xy - log_prob_x[:, np.newaxis]

        return log_prob_y_x

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate samples from fitted data.

        :param int n_samples: number of samples.
        :param Optional[int] random_state: random number generator seed.
        :return: samples.
        :rtype: np.ndarray
        """

        n_features = len(self.distributions)
        n_classes = len(self.priors)

        samples = np.zeros((n_samples, n_features, n_classes))
        for feature in range(len(self.distributions)):
            samples[:, feature, :] = self.distributions[feature].sample(  # type: ignore
                n_samples=n_samples, random_state=random_state
            )

        return samples


class GaussianNaiveBayes(ExtendedNaiveBayes):
    """
    Naive Bayes model with normal distributed features.
    """

    def __init__(self, n_features: int) -> None:
        """
        Init model with {n_features} normal distributed features.

        :param int n_features: number of features.
        """

        super().__init__(distributions=[Normal() for _ in range(n_features)])


class BernoulliNaiveBayes(ExtendedNaiveBayes):
    """
    Naive Bayes model with bernoulli distributed features.
    """

    def __init__(self, n_features: int) -> None:
        """
        Init model with {n_features} bernoulli distributed features.

        :param int n_features: number of features.
        """

        super().__init__(distributions=[Bernoulli() for _ in range(n_features)])


class CategoricalNaiveBayes(ExtendedNaiveBayes):
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


# TODO: add MultinomialNaiveBayes
