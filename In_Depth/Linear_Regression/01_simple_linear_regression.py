import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(seed=1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
x_fit = np.linspace(0, 10, 1000)
y_fit = model.predict(x_fit[:, np.newaxis])
plt.scatter(x, y, color='blue')
plt.plot(x_fit, y_fit, color='red')
plt.title(label='Simple Linear Regression \nModel slope: {0:.4f}, Model intercept: {1:.4f}'.format(model.coef_[0], model.intercept_), loc='center')
plt.show()
