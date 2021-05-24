import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from distributions import Normal
from models import NaiveBayes

np.random.seed(42)


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
            np.allclose(
                model.predict_log_proba(self.X_train),
                sklearn_model.predict_log_proba(self.X_train),
            )
        )
        self.assertTrue(
            np.allclose(
                model.predict_proba(self.X_train),
                sklearn_model.predict_proba(self.X_train),
            )
        )
        self.assertTrue(
            np.allclose(
                model.predict(self.X_train), sklearn_model.predict(self.X_train)
            )
        )
        self.assertTrue(
            np.allclose(
                model.score(self.X_train, self.y_train),
                sklearn_model.score(self.X_train, self.y_train),
            )
        )

        self.assertTrue(
            np.allclose(
                model.predict_log_proba(self.X_test),
                sklearn_model.predict_log_proba(self.X_test),
            )
        )
        self.assertTrue(
            np.allclose(
                model.predict_proba(self.X_test),
                sklearn_model.predict_proba(self.X_test),
            )
        )
        self.assertTrue(
            np.allclose(model.predict(self.X_test), sklearn_model.predict(self.X_test))
        )
        self.assertTrue(
            np.allclose(
                model.score(self.X_test, self.y_test),
                sklearn_model.score(self.X_test, self.y_test),
            )
        )
