import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as pyplot
from kmeans import KMeans

X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)

clusters = len(np.unique(y))
print('K is', clusters)

k = KMeans(K=clusters, max_iters=150, plot_steps=True)
k.fit(X_train)

y_test = k.predict(X_test)

print(y_test.shape)
print(y_test)

