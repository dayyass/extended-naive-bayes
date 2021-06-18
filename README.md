# Extended Naive Bayes

### Installation

```python3
# clone repo
git clone https://github.com/dayyass/naive_bayes.git

# install dependencies
cd naive_bayes
pip install -r requirements.txt
```

### Usage

The repository consists two modules:
1) `distributions` - contains different parametric distributions (*univariate/multivariate*, *discrete/continuous*) to fit into data;
2) `models` - contains different naive bayes models.

#### 1. Distributions

`distributions` module contains different parametric distributions (*univariate/multivariate*, *discrete/continuous*) to fit into data.

All distributions share the same interface (methods):
- `.fit(X, y)` - compute MLE given `X` (data). If y is provided, computes MLE of `X` for each class `y`;
- `.predict_log_proba(X)` - compute log probabilities given `X` (data).

> :warning: If `y` were provided to `.fit` method, then `.predict_log_proba` will compute log probabilities for each class `y`.

List of available distributions:

 Distribution | Discrete | Continuous
--- | --- | ---
**Univariate** | `Bernoulli`<br>`Binomial`<br>`Categorical`<br>`Geometric`<br>`Poisson` | `Normal`<br>`Exponential`<br>`Gamma`<br>`Beta`
**Multivariate** | `Multinomial` | `MultivariateNormal`

There are also two special kind of distributions:
- [x] `ContinuousUnivariateDistribution` - any continuous univariate distribution from scipy.stats with method `.fit` *(scipy.stats.rv_continuous.fit)* (see *example 3*);
- [x] `KernelDensityEstimator` - Kernel Density Estimation *(Parzen–Rosenblatt window method)* - non-parametric method (see *example 4*).

#### Example 1: Bernoulli distribution

```python3
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
```

#### Example 2: Normal distribution

```python3
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
```

#### Example 3: ContinuousUnivariateDistribution

```python3
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
```

#### Example 4: KernelDensityEstimator

```python3
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
```

#### 2. Models

`models` module contains different naive bayes models.

All models share the same interface (methods):
- `.fit(X, y)` - fit the model;
- `.predict(X)` - compute model predictions;
- `.predict_proba(X)` - compute class probabilities;
- `.predict_log_proba(X)` - compute class log probabilities;
- `.score(X, y)` - compute mean accuracy.

List of available models:
- [x] `NaiveBayes` - model with parameterizable feature distribution;
- [x] `BernoulliNaiveBayes` - model with Bernoulli feature distribution;
- [x] `CategoricalNaiveBayes` - model with Categorical feature distribution;
- [x] `GaussianNaiveBayes` - model with Normal feature distribution;


#### Example 1: Bernoulli distributed data

```python3
import numpy as np
from sklearn.model_selection import train_test_split

from naive_bayes import BernoulliNaiveBayes

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
```

#### Example 2: Normal distributed data

```python3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from naive_bayes import GaussianNaiveBayes

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

model = GaussianNaiveBayes(n_features=X.shape[1])
model.fit(X_train, y_train)
model.predict(X_test)
```

#### Example 3: Mix of Bernoulli and Normal distributed data

```python3
import numpy as np
from sklearn.model_selection import train_test_split

from naive_bayes import NaiveBayes
from naive_bayes.distributions import Bernoulli, Normal

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

model = NaiveBayes(
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
```

### Requirements

- `Python>=3.6`
- `numpy>=1.20.3`
- `pre-commit>=2.13.0`
- `scikit-learn>=0.24.2`

To install requirements use:<br>
`pip install -r requirements`

### Tests

All implemented distributions and models are covered with unittest.

To run tests use:<br>
`python -m unittest discover tests`

### Citations

If you use **extended_naive_bayes** in a scientific publication, we would appreciate references to the following BibTex entry:
```bibtex
@misc{dayyass2021naivebayes,
    author = {El-Ayyass, Dani},
    title = {Extension of Naive Bayes Classificator},
    howpublished = {\url{https://github.com/dayyass/extended_naive_bayes}},
    year = {2021},
}
```
