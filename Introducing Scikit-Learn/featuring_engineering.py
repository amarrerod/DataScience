# Como crear la matriz de caracteristicas a partir de los datos reales de un problema
# Categorical Features

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

# Empleamos la tecnica One-hot encoding. Crea una columna extra indicando la presencia o ausencia de una categoria con un valor de  1 o 0

vect = DictVectorizer(sparse = False, dtype = int)
print(vect.fit_transform(data))
print(vect.get_feature_names())

# Si tenemos una gran cantidad de variantes por categoria podemos usar la version de matriz dispersa para no almacenar los zeros
print("Usando una matriz de propiedades dispersa")
vect = DictVectorizer(sparse = True, dtype = int)
print(vect.fit_transform(data))

print("Trabajando con cadenas de texto")
sample = ['problem of evil',
          'evil queen',
          'horizon problem']
for i in sample:
    print("- {0}".format(i))
vect = CountVectorizer()
x = vect.fit_transform(sample)
print(x)
# Obtenemos una matriz dispersa pero sin etiquetas, para obtener las etiquetas convertimos a DataFrame
print("Buscando las etiquetas")
print(pd.DataFrame(x.toarray(), columns = vect.get_feature_names()))
print("Aplicando TF-IDF")
vect = TfidfVectorizer()
x = vect.fit_transform(sample)
print(pd.DataFrame(x.toarray(), columns = vect.get_feature_names()))
