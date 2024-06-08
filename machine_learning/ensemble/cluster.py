from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)
X, y = make_blobs(n_samples=1000, n_features=2, centers=5, random_state=42)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Original Data with True Labels")

# Fit KMeans
kmm = KMeans(n_clusters=5, n_init='auto')
kmm.fit(X)

# Plot the cluster centers
plt.scatter(kmm.cluster_centers_[:, 0], kmm.cluster_centers_[:, 1], c='red', marker='*', s=200, label='Cluster Centers')
plt.legend()
plt.title("KMeans Cluster Centers")
plt.show()