
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

sns.set()

# Testing the Simple Linear Regression
"""
    A straight-line fit to data. A model of the form y = ax + b.
"""

range = np.random.RandomState(1)
x = 10 * range.rand(150)
y = 2 * x - 5 + range.randn(150) # Here it's the model

model = LinearRegression(fit_intercept = True)
model.fit(x[:, np.newaxis], y)
x_fit = np.linspace(0, 10, 1000)
y_fit = model.predict(x_fit[:, np.newaxis])
print("Slope (Pendiente): {}".format(model.coef_[0]))
print("Intercept (Intercepcion): {}".format(model.intercept_))
plt.scatter(x, y)
plt.plot(x_fit, y_fit)
plt.show()

# Multidimensional example
x = 10 * range.rand(100, 3)
y = 0.5 + np.dot(x, [1.5, -2., 1.])
model.fit(x, y)
print("Slope (Pendiente): {}".format(model.coef_))
print("Intercept (Intercepcion): {}".format(model.intercept_))