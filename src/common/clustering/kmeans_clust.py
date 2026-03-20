"""
KMeans-based feature clustering for KitNET.

Ported from the original TensorFlow codebase (kmeans.py).
Uses sklearn KMeans on the transposed feature matrix.
"""
import numpy as np
from sklearn.cluster import KMeans as skKMeans


class KMeansClust:
    """
    KMeans clustering for feature grouping.
    Same interface as CorClust: update(x), cluster(max_clust).
    """

    def __init__(self, n: int):
        self.n = n
        self.buffer = []

    def update(self, x: np.ndarray):
        """Accumulate a single sample x of shape (n,)."""
        self.buffer.append(x)

    def cluster(self, n_clusters: int) -> list:
        """
        Cluster features into n_clusters groups using KMeans
        on the transposed feature matrix.

        Returns:
            List of lists of feature indices.
        """
        x = np.array(self.buffer)
        self.buffer = []

        n_clusters = min(n_clusters, x.shape[1])  # can't have more clusters than features
        model = skKMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(x.T)
        sk_labels = model.labels_

        feature_map = []
        for i in range(n_clusters):
            feature_map.append(np.where(sk_labels == i)[0].tolist())

        # Remove empty clusters
        feature_map = [c for c in feature_map if len(c) > 0]
        return feature_map
