# En este modelo asumimos que los datos de cada etiqueta estan distribuidos como una Distribucion Gaussiana
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
sns.set()

X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.set_title('Naive Bayes Model', size=14)
# Definimos dos distribuciones gaussianas
xlim = (-8, 8)
ylim = (-15, 5)
xg = np.linspace(xlim[0], xlim[1], 60)
yg = np.linspace(ylim[0], ylim[1], 40)
xx, yy = np.meshgrid(xg, yg)
xgrid = np.vstack([xx.ravel(), yy.ravel()]).T
# Representamos
for label, color in enumerate(['red', 'blue']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)
    Prob = np.exp(-0.5 * (xgrid - mu) ** 2 / std ** 2).prod(1)
    Prob = np.ma.masked_array(Prob, Prob < 0.03)
    ax.pcolorfast(xg, yg, Prob.reshape(xx.shape), alpha=0.5, cmap=color.title() + 's')
    ax.contour(xx, yy, Prob.reshape(xx.shape), levels =[0.01, 0.1, 0.5, 0.9], colors=color, alpha=0.2)

ax.set(xlim=xlim, ylim=ylim)
plt.show()
