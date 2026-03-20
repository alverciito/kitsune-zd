"""
Clustering algorithms for KitNET feature mapping.

Provides three interchangeable clustering strategies for the feature-mapping
phase of KitNET (Section III-A of the paper). During the FM_GRACE_PERIOD,
input features are clustered into groups; each group is assigned to one
autoencoder in the ensemble layer.

Exports:
    CorClust: Incremental correlation-distance hierarchical clustering (original Kitsune).
    DBSCANClust: DBSCAN with iterative epsilon tuning (default per Table II).
    KMeansClust: Standard KMeans clustering.
    get_clustering: Factory function to instantiate a clustering method by name.
"""
from .corclust import CorClust
from .dbscan_clust import DBSCANClust
from .kmeans_clust import KMeansClust


def get_clustering(method: str, n_features: int):
    """Factory function to create a clustering instance by method name.

    All returned instances share a common interface:
    - update(x): Accumulate a single feature vector of shape (n_features,).
    - cluster(max_clust): Return a list of feature-index lists.

    Args:
        method: Clustering algorithm name (str). One of:
            - 'corr': Incremental correlation-distance clustering (CorClust).
            - 'dbscan': DBSCAN with iterative eps tuning (DBSCANClust).
            - 'kmeans': Standard KMeans clustering (KMeansClust).
        n_features: Number of input features (int). Used to initialize
            internal data structures.

    Returns:
        Union[CorClust, DBSCANClust, KMeansClust]: Clustering instance with
            update() and cluster() methods.

    Raises:
        ValueError: If method is not one of 'corr', 'dbscan', 'kmeans'.
    """
    if method == 'corr':
        return CorClust(n_features)
    elif method == 'dbscan':
        return DBSCANClust(n_features)
    elif method == 'kmeans':
        return KMeansClust(n_features)
    else:
        raise ValueError(f"Unknown clustering method: {method}. Use 'corr', 'dbscan', or 'kmeans'.")
