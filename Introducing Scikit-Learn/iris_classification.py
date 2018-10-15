
# Entrenamiento Supervisado: Clasificacion de Iris

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Cargamos los datos
iris = sns.load_dataset('iris')
x_iris = iris.drop('species', axis = 'columns') # Eliminamos el campo especie de las columnas
y_iris = iris['species']

# Dividimos los datos en dos conjuntos: entrenamiento y testeo
xtrain, xtest, ytrain, ytest = train_test_split(x_iris, y_iris,
                                                random_state = 1)
model = GaussianNB()
print("Entrenando el Modelo GaussianNB...")
model.fit(xtrain, ytrain)
print("Evaluando nuevos datos...")
ymodel = model.predict(xtest)
print("Precision final: {}".format(accuracy_score(ytest, ymodel)))
