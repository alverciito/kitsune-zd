"""
KMeans-based feature clustering for KitNET.

Ported from the original TensorFlow codebase (kmeans.py).
Uses sklearn KMeans on the transposed feature matrix.
"""
import numpy as np
from sklearn.cluster import KMeans as skKMeans


class KMeansClust:
    """KMeans-based feature clustering for KitNET ensemble layer.

    Clusters features by running sklearn KMeans on the transposed feature
    matrix, treating each feature's time series as a sample in cluster space.
    Provides the same interface as CorClust and DBSCANClust: update(x) to
    accumulate samples, cluster(n) to produce feature groups.

    This is a simpler alternative to DBSCAN clustering, with deterministic
    convergence but no noise handling.

    Attributes:
        n: Number of features in the input data.
        buffer: List of accumulated feature vectors for batch clustering.
    """

    def __init__(self, n: int):
        """Initialize the KMeans clustering buffer.

        Args:
            n: Number of features in the input data (int).
        """
        self.n = n
        self.buffer = []

    def update(self, x: np.ndarray):
        """Accumulate a single sample for later batch clustering.

        Args:
            x: Feature vector of shape (n,) for one packet/sample.
        """
        self.buffer.append(x)

    def cluster(self, n_clusters: int) -> list:
        """Cluster features into n_clusters groups using KMeans.

        Transposes the accumulated buffer so each "sample" is a feature's
        time series, then fits KMeans with 10 random initializations. The
        buffer is cleared after clustering. If n_clusters exceeds the number
        of features, it is clamped to avoid sklearn errors.

        Args:
            n_clusters: Target number of feature groups (int). Typically
                N_FEATURES / MAX_AE_SIZE.

        Returns:
            list[list[int]]: List of feature index lists. Each inner list
                contains the indices of features assigned to one ensemble AE.
                Empty clusters are removed.
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
