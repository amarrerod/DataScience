
# Entrenamiento no supervisado: Clustering.
# Los algoritmos de clustering intentan encontrar distintos grupos dentro del conjunto de datos sin referencia a ninguna etiqueta

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture # Gaussian Mixture Model

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

# Instanciamos el nuevo modelo
model = GaussianMixture(n_components = 3, covariance_type = 'full')
model.fit(x_iris)
y_gmm = model.predict(x_iris)
# Etiquetamos
iris['PCA1'] = x2d[:, 0]
iris['PCA2'] = x2d[:, 1]
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data = iris, hue = 'species', col = 'cluster', fit_reg = False)
plt.show()