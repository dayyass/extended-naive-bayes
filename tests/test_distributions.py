import unittest

import numpy as np
from scipy import stats

from distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    ContinuousUnivariateDistribution,
    Exponential,
    Gamma,
    Gaussian,
    Geometric,
    Multinomial,
    Poisson,
)
from utils import to_categorical

np.random.seed(42)


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

        pred = dist.predict_log_proba(self.X)
        true = stats.bernoulli.logpmf(self.X, p=dist.prob)

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = stats.bernoulli.logpmf(self.X, p=dist.prob[cls])

        self.assertTrue(np.allclose(pred, true))


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

        pred = dist.predict_log_proba(self.X)
        true = stats.multinomial.logpmf(
            to_categorical(self.X, num_classes=self.k), n=1, p=dist.prob
        )

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = stats.multinomial.logpmf(
                to_categorical(self.X, num_classes=self.k), n=1, p=dist.prob[cls]
            )

        self.assertTrue(np.allclose(pred, true))


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

        pred = dist.predict_log_proba(self.X)
        true = stats.binom.logpmf(self.X, n=self.n, p=dist.prob)

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = stats.binom.logpmf(self.X, n=self.n, p=dist.prob[cls])

        self.assertTrue(np.allclose(pred, true))


class TestGeometric(unittest.TestCase):

    n = 3
    n_samples = 1000
    n_classes = 2
    X = np.random.randint(low=0, high=n, size=n_samples)
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

        pred = dist.predict_log_proba(self.X)
        true = stats.geom.logpmf(self.X, p=dist.prob)

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = stats.geom.logpmf(self.X, p=dist.prob[cls])

        self.assertTrue(np.allclose(pred, true))


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

        pred = dist.predict_log_proba(self.X)
        true = stats.poisson.logpmf(self.X, mu=dist.lambda_)

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = stats.poisson.logpmf(self.X, mu=dist.lambda_[cls])

        self.assertTrue(np.allclose(pred, true))


class TestGaussian(unittest.TestCase):

    n_samples = 1000
    n_classes = 2
    X = np.random.randn(n_samples)
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    def test_fit_X(self):
        dist = Gaussian()
        dist.fit(self.X)

        pred_1 = dist.mu
        pred_2 = dist.sigma
        true_1 = self.X.mean()
        true_2 = self.X.std()

        self.assertTrue(np.allclose(pred_1, true_1))
        self.assertTrue(np.allclose(pred_2, true_2))

    def test_predict_log_proba_X(self):
        dist = Gaussian()
        dist.fit(self.X)

        pred = dist.predict_log_proba(self.X)
        true = stats.norm.logpdf(self.X, loc=dist.mu, scale=dist.sigma)

        self.assertTrue(np.allclose(pred, true))

    def test_fit_X_y(self):
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

    def test_predict_log_proba_X_y(self):
        dist = Gaussian()
        dist.fit(self.X, self.y)

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = stats.norm.logpdf(
                self.X, loc=dist.mu[cls], scale=dist.sigma[cls]
            )

        self.assertTrue(np.allclose(pred, true))


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

        pred = dist.predict_log_proba(self.X)
        true = Exponential.logpdf(self.X, lambda_=dist.lambda_)

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = Exponential.logpdf(self.X, lambda_=dist.lambda_[cls])

        self.assertTrue(np.allclose(pred, true))

    def test_logpdf(self):
        dist = Exponential()
        dist.fit(self.X)

        pred = Exponential.logpdf(self.X, lambda_=dist.lambda_)
        true = stats.expon.logpdf(self.X, scale=1 / dist.lambda_)

        self.assertTrue(np.allclose(pred, true))


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

        pred = dist.predict_log_proba(self.X)
        true = Gamma.logpdf(self.X, alpha=dist.alpha, beta=dist.beta)

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = Gamma.logpdf(
                self.X, alpha=dist.alpha[cls], beta=dist.beta[cls]
            )

        self.assertTrue(np.allclose(pred, true))

    def test_logpdf(self):
        dist = Gamma()
        dist.fit(self.X)

        pred = Gamma.logpdf(self.X, alpha=dist.alpha, beta=dist.beta)
        true = stats.gamma.logpdf(self.X, a=dist.alpha, scale=1 / dist.beta)

        self.assertTrue(np.allclose(pred, true))


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

        pred = dist.predict_log_proba(self.X)
        true = stats.beta.logpdf(self.X, a=dist.alpha, b=dist.beta)

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = stats.beta.logpdf(
                self.X, a=dist.alpha[cls], b=dist.beta[cls]
            )

        self.assertTrue(np.allclose(pred, true))


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

        dist_2 = Gaussian()
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

        dist_2 = Gaussian()
        dist_2.fit(self.X_norm)

        pred = dist_1.predict_log_proba(self.X_norm)
        true = dist_2.predict_log_proba(self.X_norm)

        self.assertTrue(np.allclose(pred, true))

    def test_fit_gaussian_X_y(self):
        dist_1 = ContinuousUnivariateDistribution(stats.norm)
        dist_1.fit(self.X_norm, self.y)

        dist_2 = Gaussian()
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

        dist_2 = Gaussian()
        dist_2.fit(self.X_norm, self.y)

        pred = dist_1.predict_log_proba(self.X_norm)
        true = dist_2.predict_log_proba(self.X_norm)

        self.assertTrue(np.allclose(pred, true))


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

        pred = dist.predict_log_proba(self.X)
        true = stats.multinomial.logpmf(self.X, n=self.n, p=dist.prob)

        self.assertTrue(np.allclose(pred, true))

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

        pred = dist.predict_log_proba(self.X)

        true = np.zeros((self.X.shape[0], self.n_classes))
        for cls in range(self.n_classes):
            true[:, cls] = stats.multinomial.logpmf(self.X, n=self.n, p=dist.prob[cls])

        self.assertTrue(np.allclose(pred, true))
