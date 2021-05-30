import unittest

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from naive_bayes.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    ContinuousUnivariateDistribution,
    Exponential,
    Gamma,
    GaussianMixtureEstimator,
    Geometric,
    KernelDensityEstimator,
    Normal,
    Poisson,
)
from naive_bayes.distributions.multivariate import Multinomial, MultivariateNormal
from naive_bayes.utils import to_categorical

np.random.seed(42)

# TODO: add .sample tests


class TestBernoulli(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=2, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Bernoulli()
        dist.fit(self.X)

        pred = dist.prob
        true = self.X.mean()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X(self):
        dist = Bernoulli()
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.bernoulli.logpmf(self.X, p=dist.prob)
        true_2 = stats.bernoulli.pmf(self.X, p=dist.prob)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Bernoulli()
        dist.fit(self.X, self.y)

        pred = dist.prob

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = self.X[self.y == cls].mean()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X_y(self):
        dist = Bernoulli()
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.bernoulli.logpmf(self.X, p=dist.prob[cls])
            true_2[:, cls] = stats.bernoulli.pmf(self.X, p=dist.prob[cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestCategorical(unittest.TestCase):

    k = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=k, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Categorical(k=self.k)
        dist.fit(self.X)

        pred = dist.prob

        _, counts = np.unique(self.X, return_counts=True)
        true = counts / counts.sum()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X(self):
        dist = Categorical(k=self.k)
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.multinomial.logpmf(
            to_categorical(self.X, num_classes=self.k), n=1, p=dist.prob
        )
        true_2 = stats.multinomial.pmf(
            to_categorical(self.X, num_classes=self.k), n=1, p=dist.prob
        )

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Categorical(k=self.k)
        dist.fit(self.X, self.y)

        pred = dist.prob

        true = np.zeros((self.n_classes, self.k))
        for cls in range(self.n_classes):
            _, counts = np.unique(self.X[self.y == cls], return_counts=True)
            true[cls] = counts / counts.sum()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X_y(self):
        dist = Categorical(k=self.k)
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.multinomial.logpmf(
                to_categorical(self.X, num_classes=self.k), n=1, p=dist.prob[cls]
            )
            true_2[:, cls] = stats.multinomial.pmf(
                to_categorical(self.X, num_classes=self.k), n=1, p=dist.prob[cls]
            )

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestBinomial(unittest.TestCase):

    n = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=n, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Binomial(n=self.n)
        dist.fit(self.X)

        pred = dist.prob
        true = self.X.mean() / self.n

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X(self):
        dist = Binomial(n=self.n)
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.binom.logpmf(self.X, n=self.n, p=dist.prob)
        true_2 = stats.binom.pmf(self.X, n=self.n, p=dist.prob)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Binomial(n=self.n)
        dist.fit(self.X, self.y)

        pred = dist.prob

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = self.X[self.y == cls].mean() / self.n

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X_y(self):
        dist = Binomial(n=self.n)
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.binom.logpmf(self.X, n=self.n, p=dist.prob[cls])
            true_2[:, cls] = stats.binom.pmf(self.X, n=self.n, p=dist.prob[cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestGeometric(unittest.TestCase):

    n = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=1, high=n, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Geometric()
        dist.fit(self.X)

        pred = dist.prob
        true = 1 / self.X.mean()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X(self):
        dist = Geometric()
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.geom.logpmf(self.X, p=dist.prob)
        true_2 = stats.geom.pmf(self.X, p=dist.prob)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Geometric()
        dist.fit(self.X, self.y)

        pred = dist.prob

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = 1 / self.X[self.y == cls].mean()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X_y(self):
        dist = Geometric()
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.geom.logpmf(self.X, p=dist.prob[cls])
            true_2[:, cls] = stats.geom.pmf(self.X, p=dist.prob[cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestPoisson(unittest.TestCase):

    n = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=n, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Poisson()
        dist.fit(self.X)

        pred = dist.lambda_
        true = self.X.mean()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X(self):
        dist = Poisson()
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.poisson.logpmf(self.X, mu=dist.lambda_)
        true_2 = stats.poisson.pmf(self.X, mu=dist.lambda_)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Poisson()
        dist.fit(self.X, self.y)

        pred = dist.lambda_

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = self.X[self.y == cls].mean()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X_y(self):
        dist = Poisson()
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.poisson.logpmf(self.X, mu=dist.lambda_[cls])
            true_2[:, cls] = stats.poisson.pmf(self.X, mu=dist.lambda_[cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestGaussian(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.randn(n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Normal()
        dist.fit(self.X)

        pred_1 = dist.mu
        pred_2 = dist.sigma
        true_1 = self.X.mean()
        true_2 = self.X.std()

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_X(self):
        dist = Normal()
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.norm.logpdf(self.X, loc=dist.mu, scale=dist.sigma)
        true_2 = stats.norm.pdf(self.X, loc=dist.mu, scale=dist.sigma)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Normal()
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

    def test_predict_log_proba_X_y(self):
        dist = Normal()
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.norm.logpdf(
                self.X, loc=dist.mu[cls], scale=dist.sigma[cls]
            )
            true_2[:, cls] = stats.norm.pdf(
                self.X, loc=dist.mu[cls], scale=dist.sigma[cls]
            )

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestExponential(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.exponential(scale=2, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Exponential()
        dist.fit(self.X)

        pred = dist.lambda_
        true = 1 / self.X.mean()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X(self):
        dist = Exponential()
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.expon.logpdf(self.X, scale=1 / dist.lambda_)
        true_2 = stats.expon.pdf(self.X, scale=1 / dist.lambda_)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Exponential()
        dist.fit(self.X, self.y)

        pred = dist.lambda_

        true = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true[cls] = 1 / self.X[self.y == cls].mean()

        self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X_y(self):
        dist = Exponential()
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.expon.logpdf(self.X, scale=1 / dist.lambda_[cls])
            true_2[:, cls] = stats.expon.pdf(self.X, scale=1 / dist.lambda_[cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestGamma(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.exponential(scale=2, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Gamma()
        dist.fit(self.X)

        pred_1 = dist.alpha
        pred_2 = dist.beta

        true_1 = Gamma.compute_alpha_mme(self.X)
        true_2 = Gamma.compute_beta_mme(self.X)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_X(self):
        dist = Gamma()
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.gamma.logpdf(self.X, a=dist.alpha, scale=1 / dist.beta)
        true_2 = stats.gamma.pdf(self.X, a=dist.alpha, scale=1 / dist.beta)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Gamma()
        dist.fit(self.X, self.y)

        pred_1 = dist.alpha
        pred_2 = dist.beta

        true_1 = np.zeros(self.n_classes)
        true_2 = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true_1[cls] = Gamma.compute_alpha_mme(self.X[self.y == cls])
            true_2[cls] = Gamma.compute_beta_mme(self.X[self.y == cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_X_y(self):
        dist = Gamma()
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.gamma.logpdf(
                self.X, a=dist.alpha[cls], scale=1 / dist.beta[cls]
            )
            true_2[:, cls] = stats.gamma.pdf(
                self.X, a=dist.alpha[cls], scale=1 / dist.beta[cls]
            )

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestBeta(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.uniform(size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Beta()
        dist.fit(self.X)

        pred_1 = dist.alpha
        pred_2 = dist.beta

        true_1 = Beta.compute_alpha_mme(self.X)
        true_2 = Beta.compute_beta_mme(self.X)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_X(self):
        dist = Beta()
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.beta.logpdf(self.X, a=dist.alpha, b=dist.beta)
        true_2 = stats.beta.pdf(self.X, a=dist.alpha, b=dist.beta)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = Beta()
        dist.fit(self.X, self.y)

        pred_1 = dist.alpha
        pred_2 = dist.beta

        true_1 = np.zeros(self.n_classes)
        true_2 = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            true_1[cls] = Beta.compute_alpha_mme(self.X[self.y == cls])
            true_2[cls] = Beta.compute_beta_mme(self.X[self.y == cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_X_y(self):
        dist = Beta()
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.beta.logpdf(
                self.X, a=dist.alpha[cls], b=dist.beta[cls]
            )
            true_2[:, cls] = stats.beta.pdf(self.X, a=dist.alpha[cls], b=dist.beta[cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


# TODO
class TestContinuousUnivariateDistribution(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X_norm = np.random.randn(n_samples)
    # X_expon = np.random.exponential(scale=2, size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_gaussian_X(self):
        dist_1 = ContinuousUnivariateDistribution(stats.norm)
        dist_1.fit(self.X_norm)

        dist_2 = Normal()
        dist_2.fit(self.X_norm)

        pred_1 = dist_1.distribution_params[0]  # mu
        pred_2 = dist_1.distribution_params[1]  # sigma / std

        true_1 = dist_2.mu
        true_2 = dist_2.sigma

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_gaussian_X(self):
        dist_1 = ContinuousUnivariateDistribution(stats.norm)
        dist_1.fit(self.X_norm)

        dist_2 = Normal()
        dist_2.fit(self.X_norm)

        pred_1 = dist_1.predict_log_proba(self.X_norm)
        pred_2 = dist_1.predict_proba(self.X_norm)
        true_1 = dist_2.predict_log_proba(self.X_norm)
        true_2 = dist_2.predict_proba(self.X_norm)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_gaussian_X_y(self):
        dist_1 = ContinuousUnivariateDistribution(stats.norm)
        dist_1.fit(self.X_norm, self.y)

        dist_2 = Normal()
        dist_2.fit(self.X_norm, self.y)

        pred_1 = dist_1.distribution_params[:, 0]  # mu
        pred_2 = dist_1.distribution_params[:, 1]  # sigma / std

        true_1 = dist_2.mu
        true_2 = dist_2.sigma

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_gaussian_X_y(self):
        dist_1 = ContinuousUnivariateDistribution(stats.norm)
        dist_1.fit(self.X_norm, self.y)

        dist_2 = Normal()
        dist_2.fit(self.X_norm, self.y)

        pred_1 = dist_1.predict_log_proba(self.X_norm)
        pred_2 = dist_1.predict_proba(self.X_norm)
        true_1 = dist_2.predict_log_proba(self.X_norm)
        true_2 = dist_2.predict_proba(self.X_norm)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


#     # def test_fit_exponential_X(self):
#     #     dist_1 = ContinuousUnivariateDistribution(stats.gamma)
#     #     dist_1.fit(self.X_expon)
#     #
#     #     dist_2 = Exponential()
#     #     dist_2.fit(self.X_expon)
#     #
#     #     pred = dist_1.distribution_params[0]  # lambda
#     #     true = dist_2.lambda_
#     #
#     #     self.assertTrue(np.allclose(pred, true))
#     #
#     # def test_predict_log_proba_exponential_X(self):
#     #     pass
#     #
#     # def test_fit_exponential_X_y(self):
#     #     dist_1 = ContinuousUnivariateDistribution(stats.gamma)
#     #     dist_1.fit(self.X_expon, self.y)
#     #
#     #     dist_2 = Exponential()
#     #     dist_2.fit(self.X_expon, self.y)
#     #
#     #     pred = dist_1.distribution_params[:, 0]  # lambda
#     #     true = dist_2.lambda_
#     #
#     #     self.assertTrue(np.allclose(pred, true))
#     #
#     # def test_predict_log_proba_exponential_X_y(self):
#     #     pass


class TestKernelDensityEstimator(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.randn(n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    bandwidth = [0.5, 1.0, 2.0]
    kernel = ["gaussian", "linear"]

    def test_predict_log_proba_gaussian_X(self):
        for bandwidth in self.bandwidth:
            for kernel in self.kernel:
                with self.subTest():

                    dist_1 = KernelDensityEstimator(bandwidth=bandwidth, kernel=kernel)
                    dist_1.fit(self.X)

                    dist_2 = KernelDensity(bandwidth=bandwidth, kernel=kernel)
                    dist_2.fit(self.X[:, np.newaxis])

                    pred = dist_1.predict_log_proba(self.X)
                    true = dist_2.score_samples(self.X[:, np.newaxis])

                    self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_gaussian_X_y(self):
        for bandwidth in self.bandwidth:
            for kernel in self.kernel:
                with self.subTest():

                    dist_1 = KernelDensityEstimator(bandwidth=bandwidth, kernel=kernel)
                    dist_1.fit(self.X, self.y)

                    pred = dist_1.predict_log_proba(self.X)

                    true = np.zeros((self.n_samples, self.n_classes))
                    for cls in range(self.n_classes):
                        dist_2 = KernelDensity(bandwidth=bandwidth, kernel=kernel)
                        dist_2.fit(self.X[self.y == cls][:, np.newaxis])
                        true[:, cls] = dist_2.score_samples(self.X[:, np.newaxis])

                    self.assertTrue(np.allclose(pred, true))


class TestGaussianMixtureEstimator(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.randn(n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    n_components = [2, 3]
    covariance_type = ["full", "tied"]
    random_state = 42

    def test_predict_log_proba_gaussian_X(self):
        for n_components in self.n_components:
            for covariance_type in self.covariance_type:
                with self.subTest():

                    dist_1 = GaussianMixtureEstimator(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        random_state=self.random_state,
                    )
                    dist_1.fit(self.X)

                    dist_2 = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        random_state=self.random_state,
                    )
                    dist_2.fit(self.X[:, np.newaxis])

                    pred = dist_1.predict_log_proba(self.X)
                    true = dist_2.score_samples(self.X[:, np.newaxis])

                    self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_gaussian_X_y(self):
        for n_components in self.n_components:
            for covariance_type in self.covariance_type:
                with self.subTest():

                    dist_1 = GaussianMixtureEstimator(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        random_state=self.random_state,
                    )
                    dist_1.fit(self.X, self.y)

                    pred = dist_1.predict_log_proba(self.X)

                    true = np.zeros((self.n_samples, self.n_classes))
                    for cls in range(self.n_classes):
                        dist_2 = GaussianMixture(
                            n_components=n_components,
                            covariance_type=covariance_type,
                            random_state=self.random_state,
                        ).fit(self.X[self.y == cls][:, np.newaxis])
                        true[:, cls] = dist_2.score_samples(self.X[:, np.newaxis])

                    self.assertTrue(np.allclose(pred, true))


class TestMultinomial(unittest.TestCase):

    n = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.multinomial(n=n, pvals=[1 / 3, 1 / 3, 1 / 3], size=n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    # # TODO
    # def test_fit_X(self):
    #     dist = Multinomial(n=self.n)
    #     dist.fit(self.X)
    #
    #     pred = dist.prob
    #
    #     _, counts = np.unique(self.X, return_counts=True)
    #     true = counts / counts.sum()
    #
    #     self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X(self):
        dist = Multinomial(n=self.n)
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.multinomial.logpmf(self.X, n=self.n, p=dist.prob)
        true_2 = stats.multinomial.pmf(self.X, n=self.n, p=dist.prob)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    # # TODO
    # def test_fit_X_y(self):
    #     dist = Multinomial(n=self.n)
    #     dist.fit(self.X, self.y)
    #
    #     pred = dist.prob
    #
    #     true = np.zeros((self.n_classes, self.k))
    #     for cls in range(self.n_classes):
    #         _, counts = np.unique(self.X[self.y == cls], return_counts=True)
    #         true[cls] = counts / counts.sum()
    #
    #     self.assertTrue(np.allclose(pred, true))

    def test_predict_log_proba_X_y(self):
        dist = Multinomial(n=self.n)
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.multinomial.logpmf(
                self.X, n=self.n, p=dist.prob[cls]
            )
            true_2[:, cls] = stats.multinomial.pmf(self.X, n=self.n, p=dist.prob[cls])

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))


class TestMultivariateNormal(unittest.TestCase):

    n_samples = 1000
    n_features = 3
    n_classes = 2
    X = np.random.multivariate_normal(
        mean=[0 for _ in range(n_features)], cov=np.eye(n_features), size=n_samples
    )
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = MultivariateNormal()
        dist.fit(self.X)

        pred_1 = dist.mu
        pred_2 = dist.sigma

        true_1 = self.X.mean(axis=0)
        true_2 = np.cov(self.X, rowvar=False, ddof=0)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_X(self):
        dist = MultivariateNormal()
        dist.fit(self.X)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)
        true_1 = stats.multivariate_normal.logpdf(self.X, mean=dist.mu, cov=dist.sigma)
        true_2 = stats.multivariate_normal.pdf(self.X, mean=dist.mu, cov=dist.sigma)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_fit_X_y(self):
        dist = MultivariateNormal()
        dist.fit(self.X, self.y)

        pred_1 = dist.mu
        pred_2 = dist.sigma

        true_1 = np.zeros((self.n_classes, self.n_features))
        true_2 = np.zeros((self.n_classes, self.n_features, self.n_features))
        for cls in range(self.n_classes):
            true_1[cls] = self.X[self.y == cls].mean(axis=0)
            true_2[cls] = np.cov(self.X[self.y == cls], rowvar=False, ddof=0)

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_X_y(self):
        dist = MultivariateNormal()
        dist.fit(self.X, self.y)

        pred_1 = dist.predict_log_proba(self.X)
        pred_2 = dist.predict_proba(self.X)

        true_1 = np.zeros((self.n_samples, self.n_classes))
        true_2 = np.zeros((self.n_samples, self.n_classes))
        for cls in range(self.n_classes):
            true_1[:, cls] = stats.multivariate_normal.logpdf(
                self.X, mean=dist.mu[cls], cov=dist.sigma[cls]
            )
            true_2[:, cls] = stats.multivariate_normal.pdf(
                self.X, mean=dist.mu[cls], cov=dist.sigma[cls]
            )

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))
