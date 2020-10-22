import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)

clusters = len(np.unique(y))
print('K is', clusters)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X_train)

k = KMeans(init="random", n_clusters=clusters, max_iter=150)
k.fit(scaled_features)

y_test = k.predict(X_test)

print(y_test.shape)
print(y_test)

df = pd.DataFrame({"x": X_test[:,0], "y": X_test[:,1], "cluster": y_test})
print('df', df)
groups = df.groupby("cluster")

for cluster, data in groups:
    plt.scatter(data["x"], data["y"], label=cluster)
plt.show()

