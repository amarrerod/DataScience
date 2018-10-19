import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
x_fit = np.linspace(0, 10, 1000)
gauss_model = make_pipeline(GaussianFeatures(10, 1.0), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
y_fit = gauss_model.predict(x_fit[:, np.newaxis])
gf = gauss_model.named_steps['gaussianfeatures']
lm = gauss_model.named_steps['linearregression']

fig, ax = plt.subplots()
for i in range(10):
    selector = np.zeros(10)
    selector[i] = 1
    X_fit = gf.transform(x_fit[:, None]) * selector
    y_fit = lm.predict(X_fit)
    ax.fill_between(x_fit, y_fit.min(), y_fit, color='gray', alpha=0.2)

ax.scatter(x, y)
ax.plot(x_fit, gauss_model.predict(x_fit[:, np.newaxis]))
ax.set_xlim(0, 10)
ax.set_ylim(y_fit.min(), 1.5)
plt.show()
