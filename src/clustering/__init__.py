"""Clustering algorithms for KitNET feature mapping."""
from .corclust import CorClust
from .dbscan_clust import DBSCANClust
from .kmeans_clust import KMeansClust


def get_clustering(method: str, n_features: int):
    """
    Factory function to create a clustering instance.

    Args:
        method: One of 'corr', 'dbscan', 'kmeans'.
        n_features: Number of input features.

    Returns:
        Clustering instance with update() and cluster() methods.
    """
    if method == 'corr':
        return CorClust(n_features)
    elif method == 'dbscan':
        return DBSCANClust(n_features)
    elif method == 'kmeans':
        return KMeansClust(n_features)
    else:
        raise ValueError(f"Unknown clustering method: {method}. Use 'corr', 'dbscan', or 'kmeans'.")
