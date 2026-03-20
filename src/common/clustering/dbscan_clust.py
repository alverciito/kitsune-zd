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
    """DBSCAN-based feature clustering with iterative epsilon tuning.

    Clusters features by running DBSCAN on the transposed, min-max normalized
    feature matrix. The epsilon parameter is iteratively adjusted with decaying
    momentum to converge on the target number of clusters. If convergence is
    not reached within 100 iterations, the result is used as-is with a warning.

    A post-processing step ensures all clusters have at least 2 features by
    redistributing features from the largest cluster. This is the default
    clustering method per Table II of the paper (DEFAULT_CLUSTERING = 'dbscan').

    Same interface as CorClust: update(x) to accumulate samples, cluster(n)
    to produce feature groups.

    Attributes:
        n: Number of features in the input data.
        buffer: List of accumulated feature vectors for batch clustering.
    """

    def __init__(self, n: int):
        """Initialize the DBSCAN clustering buffer.

        Args:
            n: Number of features in the input data (int).
        """
        self.n = n
        self.buffer = []

    def update(self, x: np.ndarray):
        """Accumulate a single sample for later batch clustering.

        Unlike CorClust (which computes incremental statistics), DBSCANClust
        stores raw feature vectors in a buffer for batch processing.

        Args:
            x: Feature vector of shape (n,) for one packet/sample.
        """
        self.buffer.append(x)

    def cluster(self, n_clusters: int) -> list:
        """Cluster features into approximately n_clusters groups using DBSCAN.

        The algorithm works as follows:
        1. Min-max normalizes the accumulated buffer to [0, 1].
        2. Transposes so each "sample" is a feature's time series.
        3. Iteratively runs DBSCAN with adjusted eps (momentum-decayed)
           until the desired number of clusters is achieved or 100 epochs.
        4. Post-processes: shifts noise labels, ensures min cluster size of 2,
           and removes empty clusters.

        The buffer is cleared after clustering.

        Args:
            n_clusters: Target number of feature groups (int). Typically
                N_FEATURES / MAX_AE_SIZE.

        Returns:
            list[list[int]]: List of feature index lists. Each inner list
                contains the indices of features assigned to one ensemble AE.
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
