import unittest

import numpy as np

from distributions import Bernoulli

np.random.seed(42)


class TestBernoulli(unittest.TestCase):

    X = np.random.randint(low=0, high=2, size=1000)
    y = np.random.randint(low=0, high=2, size=1000)

    def test_prob_X(self):
        dist = Bernoulli()
        dist.fit(self.X)
        self.assertEqual(dist.prob, self.X.mean())

    def test_prob_X_y(self):
        dist = Bernoulli()
        dist.fit(self.X, self.y)
        self.assertTrue(
            np.allclose(
                dist.prob,
                np.array([self.X[self.y == 0].mean(), self.X[self.y == 1].mean()]),
            ),
        )
