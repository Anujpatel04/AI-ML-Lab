import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

X = df[['sepal length (cm)', 'sepal width (cm)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(X_scaled)

print("\nHierarchical Clustering Metrics:")
print("Silhouette Score:", silhouette_score(X_scaled, labels))

df['Cluster'] = labels

plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Hierarchical (Agglomerative) Clustering on Iris Dataset (2 Features)")
plt.show()

plt.figure(figsize=(10, 6))
linked = linkage(X_scaled, method='ward')

dendrogram(linked,
           orientation='top',
           distance_sort='ascending',
           show_leaf_counts=False)

plt.title("Dendrogram for Hierarchical Clustering (Ward Method)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
