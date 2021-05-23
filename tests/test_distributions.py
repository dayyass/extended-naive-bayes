import unittest

import numpy as np

from distributions import (
    Bernoulli,
    Binomial,
    Categorical,
    Exponential,
    Gaussian,
    Geometric,
    Poisson,
)

np.random.seed(42)


class TestBernoulli(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=2, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

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

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = self.X[self.y == cls].mean()

        self.assertTrue(np.allclose(pred, true))


class TestCategorical(unittest.TestCase):

    k = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=k, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

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


class TestBinomial(unittest.TestCase):

    n = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=n, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_prob_X(self):
        dist = Binomial(n=self.n)
        dist.fit(self.X)

        pred = dist.prob
        true = self.X.mean() / self.n

        self.assertTrue(np.allclose(pred, true))

    def test_prob_X_y(self):
        dist = Binomial(n=self.n)
        dist.fit(self.X, self.y)

        pred = dist.prob

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = self.X[self.y == cls].mean() / self.n

        self.assertTrue(np.allclose(pred, true))


class TestGeometric(unittest.TestCase):

    n = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=n, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_prob_X(self):
        dist = Geometric(n=self.n)
        dist.fit(self.X)

        pred = dist.prob
        true = 1 / self.X.mean()

        self.assertTrue(np.allclose(pred, true))

    def test_prob_X_y(self):
        dist = Geometric(n=self.n)
        dist.fit(self.X, self.y)

        pred = dist.prob

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = 1 / self.X[self.y == cls].mean()

        self.assertTrue(np.allclose(pred, true))


class TestPoisson(unittest.TestCase):

    n = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=n, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_prob_X(self):
        dist = Poisson()
        dist.fit(self.X)

        pred = dist.lambda_
        true = self.X.mean()

        self.assertTrue(np.allclose(pred, true))

    def test_prob_X_y(self):
        dist = Poisson()
        dist.fit(self.X, self.y)

        pred = dist.lambda_

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = self.X[self.y == cls].mean()

        self.assertTrue(np.allclose(pred, true))


class TestGaussian(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.randn(n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_prob_X(self):
        dist = Gaussian()
        dist.fit(self.X)

        pred_1 = dist.mu
        pred_2 = dist.sigma
        true_1 = self.X.mean()
        true_2 = self.X.std()

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_prob_X_y(self):
        dist = Gaussian()
        dist.fit(self.X, self.y)

        pred_1 = dist.mu
        pred_2 = dist.sigma

        true_1 = np.zeros(self.n_classes)
        true_2 = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true_1[cls] = self.X[self.y == cls].mean()
            true_2[cls] = self.X[self.y == cls].std()

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestExponential(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.randn(n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_prob_X(self):
        dist = Exponential()
        dist.fit(self.X)

        pred = dist.lambda_
        true = 1 / self.X.mean()

        self.assertTrue(np.allclose(pred, true))

    def test_prob_X_y(self):
        dist = Exponential()
        dist.fit(self.X, self.y)

        pred = dist.lambda_

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = 1 / self.X[self.y == cls].mean()

        self.assertTrue(np.allclose(pred, true))


# TODO: add Gamma tests
