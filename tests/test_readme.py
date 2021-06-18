import unittest

import numpy as np

np.random.seed(42)


class TestReadme(unittest.TestCase):
    def test_distributions_example_1(self):

        import numpy as np

        from naive_bayes.distributions import Bernoulli

        n_classes = 3
        n_samples = 100
        X = np.random.randint(low=0, high=2, size=n_samples)
        y = np.random.randint(
            low=0, high=n_classes, size=n_samples
        )  # categorical feature

        # if only X provided to fit method, then fit marginal distribution p(X)
        distribution = Bernoulli()
        distribution.fit(X)
        distribution.predict_log_proba(X)

        # if X and y provided to fit method, then fit conditional distribution p(X|y)
        distribution = Bernoulli()
        distribution.fit(X, y)
        distribution.predict_log_proba(X)

    def test_distributions_example_2(self):

        import numpy as np

        from naive_bayes.distributions import Normal

        n_classes = 3
        n_samples = 100
        X = np.random.randn(n_samples)
        y = np.random.randint(
            low=0, high=n_classes, size=n_samples
        )  # categorical feature

        # if only X provided to fit method, then fit marginal distribution p(X)
        distribution = Normal()
        distribution.fit(X)
        distribution.predict_log_proba(X)

        # if X and y provided to fit method, then fit conditional distribution p(X|y)
        distribution = Normal()
        distribution.fit(X, y)
        distribution.predict_log_proba(X)

    def test_distributions_example_3(self):

        import numpy as np
        from scipy import stats

        from naive_bayes.distributions import ContinuousUnivariateDistribution

        n_classes = 3
        n_samples = 100
        X = np.random.randn(n_samples)
        y = np.random.randint(
            low=0, high=n_classes, size=n_samples
        )  # categorical feature

        # if only X provided to fit method, then fit marginal distribution p(X)
        distribution = ContinuousUnivariateDistribution(stats.norm)
        distribution.fit(X)
        distribution.predict_log_proba(X)

        # if X and y provided to fit method, then fit conditional distribution p(X|y)
        distribution = ContinuousUnivariateDistribution(stats.norm)
        distribution.fit(X, y)
        distribution.predict_log_proba(X)

    def test_distributions_example_4(self):

        import numpy as np

        from naive_bayes.distributions import KernelDensityEstimator

        n_classes = 3
        n_samples = 100
        X = np.random.randn(n_samples)
        y = np.random.randint(
            low=0, high=n_classes, size=n_samples
        )  # categorical feature

        # if only X provided to fit method, then fit marginal distribution p(X)
        distribution = KernelDensityEstimator()
        distribution.fit(X)
        distribution.predict_log_proba(X)

        # if X and y provided to fit method, then fit conditional distribution p(X|y)
        distribution = KernelDensityEstimator()
        distribution.fit(X, y)
        distribution.predict_log_proba(X)

    def test_models_example_1(self):

        import numpy as np
        from sklearn.model_selection import train_test_split

        from naive_bayes.models import BernoulliNaiveBayes

        n_samples = 1000
        n_features = 10
        n_classes = 3

        X = np.random.randint(low=0, high=2, size=(n_samples, n_features))
        y = np.random.randint(low=0, high=n_classes, size=n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )

        model = BernoulliNaiveBayes(n_features=n_features)
        model.fit(X_train, y_train)
        model.predict(X_test)

    def test_models_example_2(self):

        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        from naive_bayes.models import GaussianNaiveBayes

        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        n_features = X.shape[1]

        model = GaussianNaiveBayes(n_features=n_features)
        model.fit(X_train, y_train)
        model.predict(X_test)

    def test_models_example_3(self):

        import numpy as np
        from sklearn.model_selection import train_test_split

        from naive_bayes.distributions import Bernoulli, Normal
        from naive_bayes.models import ExtendedNaiveBayes

        n_samples = 1000
        bernoulli_features = 3
        normal_features = 3
        n_classes = 3

        X_bernoulli = np.random.randint(
            low=0, high=2, size=(n_samples, bernoulli_features)
        )
        X_normal = np.random.randn(n_samples, normal_features)

        X = np.hstack(
            [X_bernoulli, X_normal]
        )  # shape (n_samples, bernoulli_features + normal_features)
        y = np.random.randint(low=0, high=n_classes, size=n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )

        model = ExtendedNaiveBayes(
            distributions=[
                Bernoulli(),
                Bernoulli(),
                Bernoulli(),
                Normal(),
                Normal(),
                Normal(),
            ]
        )
        model.fit(X_train, y_train)
        model.predict(X_test)
