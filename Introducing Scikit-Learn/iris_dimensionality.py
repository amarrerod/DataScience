
# Entrenamiento no supervisado
# Reducir la dimensionalidad del conjunto: iris

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set(color_codes = True)

# Cargamos los datos
iris = sns.load_dataset('iris')
x_iris = iris.drop('species', axis = 'columns') # Eliminamos el campo especie de las columnas
y_iris = iris['species']

# Creamos el modelo, PCA --> Principal Component Analysis
# Pidiendo que nos duelva solo dos caracteristicas
model = PCA(n_components = 2)
# Entrenamos el modelo
model.fit(x_iris)
# Transformamos los datos a 2-D
x2d = model.transform(x_iris)

iris['PCA1'] = x2d[:, 0]
iris['PCA2'] = x2d[:, 1]
sns.lmplot("PCA1", "PCA2", hue = 'species', data = iris, fit_reg = False)
plt.show()