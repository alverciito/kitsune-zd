"""
Incremental correlation-based feature clustering for KitNET.

Ported from the original Kitsune (NDSS'18) code by Yisroel Mirsky (MIT License).
See: https://github.com/ymirsky/Kitsune-py
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree


class CorClust:
    """
    Incrementally computes a correlation-distance matrix over streaming data,
    then clusters features into groups of at most `max_clust` features.

    Used in KitNET's feature-mapping phase to assign input features to
    individual autoencoders in the ensemble layer.
    """

    def __init__(self, n: int):
        """
        Args:
            n: Number of features in the input data.
        """
        self.n = n
        self.c = np.zeros(n)
        self.c_r = np.zeros(n)
        self.c_rs = np.zeros(n)
        self.C = np.zeros((n, n))
        self.N = 0

    def update(self, x: np.ndarray):
        """Update correlation statistics with a single sample x of shape (n,)."""
        self.N += 1
        self.c += x
        c_rt = x - self.c / self.N
        self.c_r += c_rt
        self.c_rs += c_rt ** 2
        self.C += np.outer(c_rt, c_rt)

    def corr_dist(self) -> np.ndarray:
        """Compute the current correlation distance matrix."""
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        C_rs_sqrt[C_rs_sqrt == 0] = 1e-100
        D = 1 - self.C / C_rs_sqrt
        D[D < 0] = 0
        return D

    def cluster(self, max_clust: int) -> list:
        """
        Cluster features so that no cluster has more than max_clust features.

        Returns:
            List of lists of feature indices, e.g. [[0,2,5], [1,3], [4,6,7]].
        """
        D = self.corr_dist()
        Z = linkage(D[np.triu_indices(self.n, 1)])
        max_clust = max(1, min(max_clust, self.n))
        return self._break_clust(to_tree(Z), max_clust)

    def _break_clust(self, dendro, max_clust: int) -> list:
        """Recursively break dendrogram until all clusters <= max_clust."""
        if dendro.count <= max_clust:
            return [dendro.pre_order()]
        left = self._break_clust(dendro.get_left(), max_clust)
        right = self._break_clust(dendro.get_right(), max_clust)
        return left + right
