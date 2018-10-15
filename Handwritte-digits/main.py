
# Aplicacion de las tecnicas anteriores

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_digits # Cargamos los datos
from sklearn.manifold import Isomap

componentes = 2
digits = load_digits()
# Vamos a mostrar los digitos
_, axis = plt.subplots(10, 10, figsize = (8, 8),
                        subplot_kw = {'xticks': [],
                                      'yticks': []},
                        gridspec_kw = dict(hspace = 0.1, wspace = 0.1))
for i, ax in enumerate(axis.flat):
    ax.imshow(digits.images[i], cmap = 'binary', interpolation = 'nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform = ax.transAxes, color = 'green')
plt.show()

# Para trabajar con estos datos tenemos que construir la matriz de propiedades
# matriz de tamanio [n_samples, n_features]

x = digits.data # Matriz de (1797, 64)
y = digits.target # Array de (1797)

print("Unsupervised learning: Dimensionality reduction")
# Reducir las dimensiones de  64 a 2 para poder visualizar los puntos correctamente, para ello usaremos el algoritmo Isomap
iso = Isomap(n_components = componentes)
iso.fit(digits.data)
data_red = iso.transform(digits.data)
# Ahora tenemos en data_red una matriz de [1797, 2]
# Vamos a representar los datos
plt.scatter(data_red[:, 0], data_red[:, 1], c = digits.target, edgecolors = 'none', alpha = 0.5, cmap = plt.cm.tab20)
plt.colorbar(label = 'digit label', ticks = range(10))
plt.clim(-0.5, 9.5)
plt.show()
# Empezamos el proceso de clasificacion con un algoritmo de aprendizaje supervisado
print("Classifcation on digits: Gaussian Naive Bayes Model")
# Dividimos los datos en conjuntos de entrenamiento y testeo
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 0)
model = GaussianNB()
model.fit(xtrain, ytrain)
ymodel = model.predict(xtest)
print("Accuracy score: {}".format(accuracy_score(ytest, ymodel)))
# Con este modelo tan simple obtenemos un 80 por ciento de efectividad, pero realmente no sabemos en que digitos estamos fallando mas. Para averiguar esto debemos emplear la matriz de confusion
matrix = confusion_matrix(ytest, ymodel)
sns.heatmap(matrix, square = True, annot = True, cbar = False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confussion Matrix')
plt.legend()
plt.show()
# Otra forma de ver los fallos es mostrar las entradas de nuevo pero ahora con las etiquetas que les ha dado el clasificador
_, axis = plt.subplots(10, 10, figsize = (8, 8),
                        subplot_kw = {'xticks': [],
                                      'yticks': []},
                        gridspec_kw = dict(hspace = 0.1, wspace = 0.1))
for i, ax in enumerate(axis.flat):
    ax.imshow(digits.images[i], cmap = 'binary', interpolation = 'nearest')
    ax.text(0.05, 0.05, str(ymodel[i]),
            transform = ax.transAxes, 
            color = 'green' if (ytest[i] == ymodel[i]) else 'red')
plt.show()