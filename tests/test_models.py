import unittest
from typing import Union

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB

from naive_bayes.distributions import Normal
from naive_bayes.models import (
    BernoulliNaiveBayes,
    CategoricalNaiveBayes,
    ExtendedNaiveBayes,
    GaussianNaiveBayes,
    SklearnBernoulliNaiveBayes,
    SklearnCategoricalNaiveBayes,
    SklearnExtendedNaiveBayes,
    SklearnGaussianNaiveBayes,
)
from naive_bayes.models.abstract import AbstractModel

np.random.seed(42)

# TODO: add .sample tests


def _compare_model_with_sklean(
    model: Union[AbstractModel, GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB],
    sklearn_model: Union[GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB],
    X: np.ndarray,
    y: np.ndarray,
) -> bool:
    """
    Function to compare our and sklearn models (all methods).

    :param AbstractModel model: our model.
    :param Union[GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB] sklearn_model: sklearn model.
    :param np.ndarray X: data.
    :param np.ndarray y: target.
    :return: True if all equal.
    :rtype: bool
    """

    test_1 = np.allclose(
        model.predict_log_proba(X),
        sklearn_model.predict_log_proba(X),
    )
    test_2 = np.allclose(
        model.predict_proba(X),
        sklearn_model.predict_proba(X),
    )
    test_3 = np.allclose(model.predict(X), sklearn_model.predict(X))
    test_4 = np.allclose(
        model.score(X, y),
        sklearn_model.score(X, y),
    )

    return np.all([test_1, test_2, test_3, test_4])


# TODO: add other sklearn models
class TestExtendedNaiveBayes(unittest.TestCase):

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    n_features = X.shape[1]

    def test_normal_compare_with_sklearn(self):

        # our model
        model = ExtendedNaiveBayes(
            distributions=[Normal() for _ in range(self.n_features)]
        )
        model.fit(self.X_train, self.y_train)

        # sklearn model
        sklearn_model = GaussianNB()
        sklearn_model.fit(self.X_train, self.y_train)

        self.assertTrue(
            _compare_model_with_sklean(
                model,
                sklearn_model,
                self.X_train,
                self.y_train,
            )
        )
        self.assertTrue(
            _compare_model_with_sklean(
                model,
                sklearn_model,
                self.X_test,
                self.y_test,
            )
        )


class TestGaussianNaiveBayes(unittest.TestCase):

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    n_samples, n_features = X.shape
    random_state = 42

    def test_compare_with_sklearn(self):

        # sklearn model
        sklearn_model = GaussianNB()
        sklearn_model.fit(self.X_train, self.y_train)

        for model_class in [GaussianNaiveBayes, SklearnGaussianNaiveBayes]:
            with self.subTest():

                # our model
                if model_class == GaussianNaiveBayes:
                    model = model_class(n_features=self.n_features)
                elif model_class == SklearnGaussianNaiveBayes:
                    model = model_class()

                model.fit(self.X_train, self.y_train)

                self.assertTrue(
                    _compare_model_with_sklean(
                        model,
                        sklearn_model,
                        self.X_train,
                        self.y_train,
                    )
                )
                self.assertTrue(
                    _compare_model_with_sklean(
                        model,
                        sklearn_model,
                        self.X_test,
                        self.y_test,
                    )
                )

                self.assertTrue(np.allclose(model.theta_, sklearn_model.theta_))
                self.assertTrue(np.allclose(model.sigma_, sklearn_model.sigma_))

    def test_compare_sampling(self):

        model_1 = GaussianNaiveBayes(n_features=self.n_features)
        model_1.fit(self.X_train, self.y_train)

        model_2 = SklearnGaussianNaiveBayes()
        model_2.fit(self.X_train, self.y_train)

        self.assertTrue(
            np.allclose(
                model_1.sample(
                    n_samples=self.n_samples, random_state=self.random_state
                ),
                model_2.sample(
                    n_samples=self.n_samples, random_state=self.random_state
                ),
            )
        )


# TODO: improve the model to work with alpha > 0
class TestBernoulliNaiveBayes(unittest.TestCase):

    n_samples = 1000
    n_features = 5
    n_classes = 2
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.random.randint(low=0, high=n_classes, size=n_samples)
    random_state = 42

    def test_compare_with_sklearn(self):

        # sklearn model
        sklearn_model = BernoulliNB(alpha=0)
        sklearn_model.fit(self.X, self.y)

        for model_class in [BernoulliNaiveBayes, SklearnBernoulliNaiveBayes]:
            with self.subTest():

                # our model
                if model_class == BernoulliNaiveBayes:
                    model = model_class(n_features=self.n_features)
                elif model_class == SklearnBernoulliNaiveBayes:
                    model = model_class(alpha=0)

                model.fit(self.X, self.y)

                self.assertTrue(
                    _compare_model_with_sklean(
                        model,
                        sklearn_model,
                        self.X,
                        self.y,
                    )
                )

                self.assertTrue(
                    np.allclose(
                        model.feature_log_prob_, sklearn_model.feature_log_prob_
                    )
                )

    def test_compare_sampling(self):

        model_1 = BernoulliNaiveBayes(n_features=self.n_features)
        model_1.fit(self.X, self.y)

        model_2 = SklearnBernoulliNaiveBayes(alpha=0)
        model_2.fit(self.X, self.y)

        self.assertTrue(
            np.allclose(
                model_1.sample(
                    n_samples=self.n_samples, random_state=self.random_state
                ),
                model_2.sample(
                    n_samples=self.n_samples, random_state=self.random_state
                ),
            )
        )


# TODO: improve the model to work with alpha > 0
class TestCategoricalNaiveBayes(unittest.TestCase):

    n_samples = 1000
    n_features = 5
    n_categories = [10, 6, 8, 3, 4]
    n_classes = 2
    X = np.random.randint(n_categories, size=(n_samples, n_features))
    y = np.random.randint(low=0, high=n_classes, size=n_samples)
    random_state = 42

    def test_compare_with_sklearn(self):

        # sklearn model
        sklearn_model = CategoricalNB(alpha=0)
        sklearn_model.fit(self.X, self.y)

        for model_class in [CategoricalNaiveBayes, SklearnCategoricalNaiveBayes]:
            with self.subTest():

                # our model
                if model_class == CategoricalNaiveBayes:
                    model = model_class(
                        n_features=self.n_features, n_categories=self.n_categories
                    )
                elif model_class == SklearnCategoricalNaiveBayes:
                    model = model_class(alpha=0)

                model.fit(self.X, self.y)

                self.assertTrue(
                    _compare_model_with_sklean(
                        model,
                        sklearn_model,
                        self.X,
                        self.y,
                    )
                )

                self.assertTrue(
                    np.all(
                        [
                            np.allclose(
                                model.feature_log_prob_[feature],
                                sklearn_model.feature_log_prob_[feature],
                            )
                            for feature in range(self.n_features)
                        ]
                    )
                )

    def test_compare_sampling(self):

        model_1 = CategoricalNaiveBayes(
            n_features=self.n_features, n_categories=self.n_categories
        )
        model_1.fit(self.X, self.y)

        model_2 = SklearnCategoricalNaiveBayes(alpha=0)
        model_2.fit(self.X, self.y)

        self.assertTrue(
            np.allclose(
                model_1.sample(
                    n_samples=self.n_samples, random_state=self.random_state
                ),
                model_2.sample(
                    n_samples=self.n_samples, random_state=self.random_state
                ),
            )
        )


class TestSklearnExtendedNaiveBayes(unittest.TestCase):

    n_samples = 1000
    n_features = 5
    n_classes = 2
    n_categories = [10, 6, 8, 3, 4]

    def test_normal_compare_with_sklearn(self):

        # data
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        n_features = X.shape[1]

        # our model
        model = SklearnExtendedNaiveBayes(
            distributions=["gaussian" for _ in range(n_features)]
        )
        self.assertTrue(set(model.models.keys()) == {"gaussian"})
        model.fit(X_train, y_train)

        # sklearn model
        sklearn_model = GaussianNB()
        sklearn_model.fit(X_train, y_train)

        self.assertTrue(
            _compare_model_with_sklean(
                model,
                sklearn_model,
                X_train,
                y_train,
            )
        )
        self.assertTrue(
            _compare_model_with_sklean(
                model,
                sklearn_model,
                X_test,
                y_test,
            )
        )

    def test_bernoulli_compare_with_sklearn(self):

        # data
        X = np.random.randint(2, size=(self.n_samples, self.n_features))
        y = np.random.randint(low=0, high=self.n_classes, size=self.n_samples)

        # our model
        model = SklearnExtendedNaiveBayes(
            distributions=["bernoulli" for _ in range(self.n_features)]
        )
        self.assertTrue(set(model.models.keys()) == {"bernoulli"})
        model.fit(X, y)

        # sklearn model
        # TODO: improve the model to work with alpha > 0
        sklearn_model = BernoulliNB(alpha=0)
        sklearn_model.fit(X, y)

        self.assertTrue(_compare_model_with_sklean(model, sklearn_model, X, y))

    def test_categorical_compare_with_sklearn(self):

        # data
        X = np.random.randint(self.n_categories, size=(self.n_samples, self.n_features))
        y = np.random.randint(low=0, high=self.n_classes, size=self.n_samples)

        # our model
        model = SklearnExtendedNaiveBayes(
            distributions=["categorical" for _ in range(self.n_features)]
        )
        self.assertTrue(set(model.models.keys()) == {"categorical"})
        model.fit(X, y)

        # sklearn model
        # TODO: improve the model to work with alpha > 0
        sklearn_model = CategoricalNB(alpha=0)
        sklearn_model.fit(X, y)

        self.assertTrue(_compare_model_with_sklean(model, sklearn_model, X, y))

    def test_normal_bernoulli_categorucal_compare_with_sklearn(self):

        # data
        X_normal = np.random.randn(self.n_samples, self.n_features)
        X_bernoulli = np.random.randint(2, size=(self.n_samples, self.n_features))
        X_categorical = np.random.randint(
            self.n_categories, size=(self.n_samples, self.n_features)
        )

        X = np.hstack([X_normal, X_bernoulli, X_categorical])
        y = np.random.randint(low=0, high=self.n_classes, size=self.n_samples)

        # our model
        model = SklearnExtendedNaiveBayes(
            distributions=X_normal.shape[1] * ["gaussian"]
            + X_bernoulli.shape[1] * ["bernoulli"]
            + X_categorical.shape[1] * ["categorical"]
        )
        self.assertTrue(
            set(model.models.keys()) == {"gaussian", "bernoulli", "categorical"}
        )
        model.fit(X, y)

        # sklearn models
        sklearn_normal_model = GaussianNB()
        sklearn_normal_model.fit(X_normal, y)

        sklearn_bernoulli_model = BernoulliNB(alpha=0)  # TODO: fix hardcore
        sklearn_bernoulli_model.fit(X_bernoulli, y)

        sklearn_categorical_model = CategoricalNB(alpha=0)  # TODO: fix hardcore
        sklearn_categorical_model.fit(X_categorical, y)

        pred_log_prob = model.predict_log_proba(X)
        true_log_prob = (
            sklearn_normal_model.predict_log_proba(X_normal)
            + sklearn_bernoulli_model.predict_log_proba(X_bernoulli)
            + sklearn_categorical_model.predict_log_proba(X_categorical)
        )

        self.assertTrue(np.allclose(pred_log_prob, true_log_prob))
