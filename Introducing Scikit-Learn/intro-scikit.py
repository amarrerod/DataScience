
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
x_iris = iris.drop('species', axis = 'columns') # Eliminamos el campo especie de las columnas
y_iris = iris['species']
print("Features matrix: {}".format(x_iris.shape))
print("Target array: {}".format(y_iris.shape))
