
# Ejemplo de aprendizaje supervisado: Regresion Lineal Simple

import matplotlib.pyplot as plt
import numpy as np

# Escogemos el modelo que vamos a emplear
from sklearn.linear_model import LinearRegression

range = np.random.RandomState(42)
x = 10 * range.rand(50)
y = 2 * x - 1 + range.rand(50)
# plt.scatter(x, y)
# plt.show()

# Ahora con estos datos empezamos a trabajar
# Una vez importada la clase del modelo lo instanciamos

model = LinearRegression(fit_intercept = True)
# Reorganizamos los datos en una matriz de propiedades
x = x[:, np.newaxis]
# Entrenamos el modelo
print("Entrenando el modelo...")
model.fit(x, y)
# Cuando se termine el entrenamiento, en el modelo se guardan los valores a y b
print("Slope (a - coef): {}".format(model.coef_))
print("Intercept (b): {}".format(model.intercept_))
# Ahora que el modelo esta entrenado, se puede emplear para predecir nuevos datos
print("Creando nuevos datos...")
xfit = np.linspace(-1, 11)
xfit = xfit[:, np.newaxis]
yfit = model.predict(xfit)
print("Resultado: ")
plt.scatter(x, y, label = "Entrenamiento", color = 'r')
plt.plot(xfit, yfit, label = "Testeo", color = 'blue')
plt.title("Ejemplo de aprendizaje supervisado: Regresion Lineal Simple")
plt.legend()
plt.show()
