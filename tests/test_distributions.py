import unittest

import numpy as np

from distributions import Bernoulli, Categorical

np.random.seed(42)


class TestBernoulli(unittest.TestCase):

    X = np.random.randint(low=0, high=2, size=1000)
    y = np.random.randint(low=0, high=2, size=1000)

    def test_prob_X(self):
        dist = Bernoulli()
        dist.fit(self.X)

        pred = dist.prob
        true = self.X.mean()

        self.assertEqual(pred, true)

    def test_prob_X_y(self):
        dist = Bernoulli()
        dist.fit(self.X, self.y)

        pred = dist.prob
        true = np.array([self.X[self.y == 0].mean(), self.X[self.y == 1].mean()])
        self.assertTrue(np.allclose(pred, true))


class TestCategorical(unittest.TestCase):

    k = 3
    n_classes = 2
    X = np.random.randint(low=0, high=k, size=1000)
    y = np.random.randint(low=0, high=n_classes, size=1000)

    def test_prob_X(self):
        dist = Categorical(k=self.k)
        dist.fit(self.X)

        pred = dist.prob

        _, counts = np.unique(self.X, return_counts=True)
        true = counts / counts.sum()

        self.assertTrue(np.allclose(pred, true))

    def test_prob_X_y(self):
        dist = Categorical(k=self.k)
        dist.fit(self.X, self.y)

        pred = dist.prob

        true = np.zeros((self.n_classes, self.k))
        for cls in range(self.n_classes):
            _, counts = np.unique(self.X[self.y == cls], return_counts=True)
            true[cls] = counts / counts.sum()

        self.assertTrue(np.allclose(pred, true))
