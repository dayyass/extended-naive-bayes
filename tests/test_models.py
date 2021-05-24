import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, _BaseNB

from distributions import Normal
from models import GaussianNaiveBayes, NaiveBayes
from models.abstract import AbstractModel

np.random.seed(42)


def compare_model_with_sklean(
    model: AbstractModel,
    sklearn_model: _BaseNB,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> bool:
    """
    Function to compare our and sklearn models (all methods).

    :param AbstractModel model: our model.
    :param _BaseNB sklearn_model: sklearn model.
    :param np.ndarray X_train: training data.
    :param np.ndarray X_test: testing data.
    :param np.ndarray y_train: training target.
    :param np.ndarray y_test: testing target.
    :return: True if all equal.
    :rtype: bool
    """

    test_1 = np.allclose(
        model.predict_log_proba(X_train),
        sklearn_model.predict_log_proba(X_train),
    )
    test_2 = np.allclose(
        model.predict_proba(X_train),
        sklearn_model.predict_proba(X_train),
    )
    test_3 = np.allclose(model.predict(X_train), sklearn_model.predict(X_train))
    test_4 = np.allclose(
        model.score(X_train, y_train),
        sklearn_model.score(X_train, y_train),
    )
    test_5 = np.allclose(
        model.predict_log_proba(X_test),
        sklearn_model.predict_log_proba(X_test),
    )
    test_6 = np.allclose(
        model.predict_proba(X_test),
        sklearn_model.predict_proba(X_test),
    )
    test_7 = np.allclose(model.predict(X_test), sklearn_model.predict(X_test))
    test_8 = np.allclose(
        model.score(X_test, y_test),
        sklearn_model.score(X_test, y_test),
    )

    return np.all([test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8])


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
            compare_model_with_sklean(
                model,
                sklearn_model,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            )
        )


class TestGaussianNaiveBayes(unittest.TestCase):

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )

    def test_fit_normal_sklearn(self):

        # our model
        model = GaussianNaiveBayes(n_features=self.X_train.shape[1])
        model.fit(self.X_train, self.y_train)

        # sklearn model
        sklearn_model = GaussianNB()
        sklearn_model.fit(self.X_train, self.y_train)

        self.assertTrue(
            compare_model_with_sklean(
                model,
                sklearn_model,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            )
        )
