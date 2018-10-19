import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
sns.set()

data = fetch_20newsgroups()
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)
matrix = confusion_matrix(test.target, labels)
sns.heatmap(matrix.T, annot=True, fmt="d", cbar=False, square=True, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()
