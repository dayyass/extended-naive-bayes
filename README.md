# Naive Bayes

### Overview
TODO

### Installation
```python3
# clone repo
git clone https://github.com/dayyass/naive_bayes.git

# install dependencies
cd naive_bayes
pip install -r requirements.txt
```

### Usage
#### Example 1: Normal distributed data
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

#### Example 2: Bernoulli distributed data
```python3
import numpy as np
from sklearn.model_selection import train_test_split

from naive_bayes import BernoulliNaiveBayes


n_samples = 1000
n_features = 10
n_classes = 2

X = np.random.randint(2, size=(n_samples, n_features))
y = np.random.randint(low=0, high=n_classes, size=n_samples)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

model = BernoulliNaiveBayes(n_features=n_features)
model.fit(X_train, y_train)
model.predict(X_test)
```

#### Example 3: Categorical distributed data
```python3
import numpy as np
from sklearn.model_selection import train_test_split

from naive_bayes import CategoricalNaiveBayes


n_samples = 1000
n_features = 5
n_categories = [10, 6, 8, 3, 4]
n_classes = 2

X = np.random.randint(n_categories, size=(n_samples, n_features))
y = np.random.randint(low=0, high=n_classes, size=n_samples)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

model = CategoricalNaiveBayes(
    n_features=n_features, n_categories=n_categories
)
model.fit(X_train, y_train)
model.predict(X_test)
```
