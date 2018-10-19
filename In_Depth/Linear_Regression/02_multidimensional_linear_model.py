import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(seed=1)
x = 10 * rng.rand(100, 3)
y = .5 + np.dot(x, [1.5, -2., 1.])
print("X: {}".format(x))
print("Y: {}".format(y))
model = LinearRegression(fit_intercept=True)
model.fit(x, y)
print('Model slope: {}'.format(model.coef_))
print('Model intercept: {0:.4f}'.format(model.intercept_))
