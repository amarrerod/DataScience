import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2, verbose=False):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    if verbose:
        print("Initial centers: \n{}".format(centers))

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        if verbose:
            print("Labels: \n{}".format(labels))
            print("New centers: \n{}".format(new_centers))
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

# Creamos un conjunto de datos bi-dimensional con cuatro centros
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
seeds = [i for i in range(0, 6)]
cmaps = ['Pastel1', 'Paired', 'viridis', 'Accent', 'Set1', 'tab20']
figure, axes = plt.subplots(2, 3, sharex='row', sharey='col')
j, k = 0, 0
for i in range(len(seeds)):
    centers, labels = find_clusters(X, n_clusters=4, rseed=seeds[i], verbose=False)
    axes[j, k].scatter(X[:, 0], X[:, 1], c = labels, s = 50, cmap=cmaps[i])
    axes[j, k].set_title("Resultados con seed: {}".format(seeds[i]))
    if i == 2:
        k = 0
        j += 1
    else:
        k += 1

plt.legend(loc='best')
plt.show()
