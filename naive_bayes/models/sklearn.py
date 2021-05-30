from typing import List, Optional

import numpy as np
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB

from naive_bayes.models.abstract import AbstractModel


class SklearnExtendedNaiveBayes(AbstractModel):
    """
    Extended (allow different distributions for each feature) Naive Bayes model based on sklearn models.
    Model works as a combination of GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB.
    """

    # TODO: fix alpha=0
    def __init__(
        self,
        distributions: List[str],
        gaussian_model: GaussianNB = GaussianNB(),
        bernoulli_model: BernoulliNB = BernoulliNB(alpha=0),
        categorical_model: CategoricalNB = CategoricalNB(alpha=0),
        multinomial_model: MultinomialNB = MultinomialNB(),
    ) -> None:
        """
        Init model with distribution for each feature.

        :param List[str] distributions: list of feature distributions. Must be one of:
           ‘gaussian’
              normal distributed feature (sklearn.naive_bayes.GaussianNB is used)
           ‘bernoulli’
              bernoulli distributed feature (sklearn.naive_bayes.BernoulliNB is used)
           ‘categorical’
              categorical distributed feature (sklearn.naive_bayes.CategoricalNB is used)
           ‘multinomial’
              multinomial distributed feature (sklearn.naive_bayes.MultinomialNB is used)
        :param GaussianNB gaussian_model: use to specify GaussianNB parameters (if you don't want to use defaults)
        :param BernoulliNB bernoulli_model: use to specify BernoulliNB parameters (if you don't want to use defaults)
        :param CategoricalNB categorical_model: use to specify CategoricalNB parameters (if you don't want to use defaults)
        :param MultinomialNB multinomial_model: use to specify MultinomialNB parameters (if you don't want to use defaults)
        """

        super().__init__(distributions)  # type: ignore
        self.models = {
            "gaussian": gaussian_model,
            "bernoulli": bernoulli_model,
            "categorical": categorical_model,
            "multinomial": multinomial_model,
        }
        self._permitted_distributions = set(self.models.keys())

        unique_distributions = np.unique(distributions)
        self._check_input_distributions(unique_distributions)

        # remove unused models
        for model_name in self._permitted_distributions:
            if model_name not in unique_distributions:
                self.models.pop(model_name)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> None:
        """
        Method to fit the model.

        :param np.ndarray X: training data.
        :param np.ndarray y: target values.
        :param Optional[np.ndarray] sample_weight: Weights applied to individual samples (1. for unweighted).
        """

        self._check_input_data(X=X, y=y)
        self._check_sample_weight(sample_weight)

        for distribution in self.models.keys():
            features_idx = self._get_features_idx_given_distribution_name(distribution)
            self.models[distribution].fit(
                X[:, features_idx], y, sample_weight=sample_weight
            )

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Method to partially fit the model.

        :param np.ndarray X: training data.
        :param np.ndarray y: target values.
        :param Optional[np.ndarray] classes: List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted in subsequent calls.
        :param Optional[np.ndarray] sample_weight: Weights applied to individual samples (1. for unweighted).
        """

        self._check_input_data(X=X, y=y)
        self._check_sample_weight(sample_weight)

        for distribution in self.models.keys():
            features_idx = self._get_features_idx_given_distribution_name(distribution)
            self.models[distribution].partial_fit(
                X[:, features_idx], y, classes=classes, sample_weight=sample_weight
            )

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Method to compute class log probabilities.

        :param np.ndarray X: training data.
        :return: class log probabilities.
        :rtype: np.ndarray
        """

        self._check_input_data(X=X)

        n_classes = list(self.models.values())[0].class_count_.shape[0]
        n_samples = X.shape[0]

        log_prob_y_x = np.zeros((n_samples, n_classes))

        for distribution, model in self.models.items():
            features_idx = self._get_features_idx_given_distribution_name(distribution)
            log_prob_y_x += model.predict_log_proba(X[:, features_idx])

        return log_prob_y_x

    # TODO: maybe use ones and save mapping into dictionary for optimization
    def _get_features_idx_given_distribution_name(
        self, distribution: str
    ) -> np.ndarray:
        """
        Select feature indices according to given distribution names.

        :param str distribution: distribution name.
        :return: feature indices.
        :rtype: np.ndarray
        """

        features_idx = np.where(np.array(self.distributions) == distribution)[0]
        return features_idx

    def _check_input_distributions(self, distributions: List[str]) -> None:
        """
        Method to check correctness of input distributions.

        :param List[str] distributions: list of feature distributions
        """

        correct = np.all(
            [(dist in self._permitted_distributions) for dist in distributions]
        )
        assert (
            correct
        ), f"feature distribution must be one of the following: {self._permitted_distributions}"

    def _check_sample_weight(self, sample_weight: Optional[np.ndarray]) -> None:
        """
        Method to check correctness of sample_weight.

        :param np.ndarray sample_weight: Weights applied to individual samples (1. for unweighted).
        """
        if sample_weight:
            assert sample_weight.ndim == 1, "sample_weight should be a 1d vector."
