"""
DBSCAN-based feature clustering for KitNET.

Ported from the original TensorFlow codebase (dbscan.py).
Uses sklearn DBSCAN with iterative eps tuning to achieve
a target number of clusters.
"""
import logging
import numpy as np
from sklearn.cluster import DBSCAN as skDBSCAN

log = logging.getLogger(__name__)


class DBSCANClust:
    """
    DBSCAN clustering that iteratively adjusts eps to produce
    approximately n_clusters groups. Falls back to splitting
    the largest cluster if convergence isn't reached.

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
        Cluster features into n_clusters groups using DBSCAN on
        the transposed feature matrix.

        Returns:
            List of lists of feature indices.
        """
        x = np.array(self.buffer)
        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-16)
        self.buffer = []

        current_clusters = -1
        eps = 1.4
        sk_labels = None
        momentum = 1.0
        epoch = 0

        while current_clusters != n_clusters and epoch < 100:
            model = skDBSCAN(eps=eps, min_samples=8).fit(x.T)
            sk_labels = model.labels_
            current_clusters = len(set(sk_labels))
            diff = n_clusters - current_clusters
            eps += 0.1 * diff * momentum
            momentum *= 0.99
            if eps < 0:
                eps = 0.05
            epoch += 1
            if epoch == 100:
                log.warning(f'DBSCAN: Could not converge to {n_clusters} clusters. '
                            f'Got {current_clusters}.')

        # Shift labels so noise (-1) becomes 0
        sk_labels = np.array(sk_labels) + 1

        feature_map = []
        for i in range(n_clusters):
            feature_map.append(np.where(sk_labels == i)[0].tolist())

        # Hot-fix: ensure minimum cluster size of 2
        cluster_lens = [len(c) for c in feature_map]
        while min(cluster_lens) < 2 and max(cluster_lens) > 4:
            idx_max = np.argmax(cluster_lens)
            idx_min = np.argmin(cluster_lens)
            feature_map[idx_min].append(feature_map[idx_max].pop(-1))
            cluster_lens = [len(c) for c in feature_map]

        # Remove empty clusters
        feature_map = [c for c in feature_map if len(c) > 0]

        return feature_map
