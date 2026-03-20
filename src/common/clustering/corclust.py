"""
Incremental correlation-based feature clustering for KitNET.

Ported from the original Kitsune (NDSS'18) code by Yisroel Mirsky (MIT License).
See: https://github.com/ymirsky/Kitsune-py
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree


class CorClust:
    """Incremental correlation-based feature clustering for KitNET.

    Incrementally computes a correlation-distance matrix over streaming data,
    then clusters features into groups of at most ``max_clust`` features using
    hierarchical agglomerative clustering with complete linkage.

    Used in KitNET's feature-mapping phase (FM_GRACE_PERIOD) to assign input
    features to individual autoencoders in the ensemble layer. This is the
    original clustering method from the Kitsune (NDSS'18) paper.

    The incremental statistics (Welford-like) allow processing one sample at
    a time without storing the full dataset in memory.

    Attributes:
        n: Number of features.
        c: Running sum of feature values, shape (n,).
        c_r: Running sum of centered residuals, shape (n,).
        c_rs: Running sum of squared centered residuals, shape (n,).
        C: Running outer product of centered residuals, shape (n, n).
        N: Number of samples observed.
    """

    def __init__(self, n: int):
        """Initialize incremental correlation statistics.

        Args:
            n: Number of features in the input data (int).
        """
        self.n = n
        self.c = np.zeros(n)
        self.c_r = np.zeros(n)
        self.c_rs = np.zeros(n)
        self.C = np.zeros((n, n))
        self.N = 0

    def update(self, x: np.ndarray):
        """Update incremental correlation statistics with a single sample.

        Uses a Welford-like online algorithm to maintain running sums for
        computing the correlation matrix without storing all samples.

        Args:
            x: Feature vector of shape (n,) for one packet/sample.
        """
        self.N += 1
        self.c += x
        c_rt = x - self.c / self.N
        self.c_r += c_rt
        self.c_rs += c_rt ** 2
        self.C += np.outer(c_rt, c_rt)

    def corr_dist(self) -> np.ndarray:
        """Compute the pairwise correlation distance matrix from accumulated statistics.

        Correlation distance is defined as D(i,j) = 1 - corr(i,j), where
        corr(i,j) is the Pearson correlation between features i and j.
        Values are clipped to [0, inf) to handle numerical imprecision.

        Returns:
            np.ndarray: Symmetric distance matrix of shape (n, n) where
                D[i,j] = 0 means features i and j are perfectly correlated,
                and D[i,j] = 1 means they are uncorrelated.
        """
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        C_rs_sqrt[C_rs_sqrt == 0] = 1e-100
        D = 1 - self.C / C_rs_sqrt
        D[D < 0] = 0
        return D

    def cluster(self, max_clust: int) -> list:
        """Cluster features into groups of at most max_clust features.

        Computes the correlation distance matrix, applies hierarchical
        agglomerative clustering (scipy linkage), and recursively splits
        the dendrogram until all clusters have at most max_clust features.

        Args:
            max_clust: Maximum number of features allowed per cluster (int).
                Corresponds to MAX_AE_SIZE in config (default 4 per Table II).

        Returns:
            list[list[int]]: List of feature index lists, e.g.,
                [[0, 2, 5], [1, 3], [4, 6, 7]]. Each inner list is the set
                of feature indices assigned to one ensemble autoencoder.
        """
        if self.n <= 1:
            return [list(range(self.n))]
        D = self.corr_dist()
        Z = linkage(D[np.triu_indices(self.n, 1)])
        max_clust = max(1, min(max_clust, self.n))
        return self._break_clust(to_tree(Z), max_clust)

    def _break_clust(self, dendro, max_clust: int) -> list:
        """Recursively split a dendrogram node until all leaf clusters have at most max_clust features.

        Args:
            dendro: scipy.cluster.hierarchy.ClusterNode to split.
            max_clust: Maximum cluster size (int).

        Returns:
            list[list[int]]: Flat list of feature-index clusters.
        """
        if dendro.count <= max_clust:
            return [dendro.pre_order()]
        left = self._break_clust(dendro.get_left(), max_clust)
        right = self._break_clust(dendro.get_right(), max_clust)
        return left + right
