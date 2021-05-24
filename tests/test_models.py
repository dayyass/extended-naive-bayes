import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, _BaseNB

from naive_bayes.distributions import Normal
from naive_bayes.models import (
    BernoulliNaiveBayes,
    CategoricalNaiveBayes,
    GaussianNaiveBayes,
    NaiveBayes,
)
from naive_bayes.models.abstract import AbstractModel

np.random.seed(42)


def _compare_model_with_sklean(
    model: AbstractModel,
    sklearn_model: _BaseNB,
    X: np.ndarray,
    y: np.ndarray,
) -> bool:
    """
    Function to compare our and sklearn models (all methods).

    :param AbstractModel model: our model.
    :param _BaseNB sklearn_model: sklearn model.
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
class TestNaiveBayes(unittest.TestCase):

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )

    def test_fit_normal_sklearn(self):

        # our model
        model = NaiveBayes(distributions=[Normal(), Normal(), Normal(), Normal()])
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
        X, y, test_size=0.5, random_state=0
    )

    def test_fit_sklearn(self):

        # our model
        model = GaussianNaiveBayes(n_features=self.X_train.shape[1])
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


# TODO: improve the model to work with alpha > 0
class TestBernoulliNaiveBayes(unittest.TestCase):

    n_samples = 1000
    n_features = 10
    n_classes = 2
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_sklearn(self):

        # our model
        model = BernoulliNaiveBayes(n_features=self.n_features)
        model.fit(self.X, self.y)

        # sklearn model
        sklearn_model = BernoulliNB(alpha=0)
        sklearn_model.fit(self.X, self.y)

        self.assertTrue(
            _compare_model_with_sklean(
                model,
                sklearn_model,
                self.X,
                self.y,
            )
        )


# TODO: improve the model to work with alpha > 0
class TestCategoricalNaiveBayes(unittest.TestCase):

    n_samples = 1000
    n_features = 5
    n_categories = [10, 6, 8, 3, 4]
    n_classes = 2
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_sklearn(self):

        # our model
        model = CategoricalNaiveBayes(
            n_features=self.n_features, n_categories=self.n_categories
        )
        model.fit(self.X, self.y)

        # sklearn model
        sklearn_model = CategoricalNB(alpha=0)
        sklearn_model.fit(self.X, self.y)

        self.assertTrue(
            _compare_model_with_sklean(
                model,
                sklearn_model,
                self.X,
                self.y,
            )
        )
