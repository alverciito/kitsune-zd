"""
CentroidDetector: K-means++ based anomaly detector.

From IEEE paper Sec IV-D:
  Training: K-means++ on Detection Frame to compute K centroids.
  Execution: distance to nearest centroid as anomaly score D(x,t).
  Paired with mean filter for smoothing.
"""
import numpy as np
from sklearn.cluster import KMeans
from .filters import mean_filter


class CentroidDetector:
    """
    Detect anomalies by measuring distance to learned cluster centroids.

    Args:
        n_clusters: Number of centroids (K) for K-means++.
        filter_window: Window size for mean filter smoothing (0 = no filter).
    """

    def __init__(self, n_clusters: int = 8, filter_window: int = 100):
        self.n_clusters = n_clusters
        self.filter_window = filter_window
        self.kmeans = None

    def train(self, scores: np.ndarray):
        """
        Fit K-means++ on the detection frame scores.

        Args:
            scores: 1D array of anomaly scores from training phase.
        """
        X = scores.reshape(-1, 1)
        self.kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(X)),
            init='k-means++',
            n_init=10,
            random_state=1234,
        )
        self.kmeans.fit(X)

    def execute(self, scores: np.ndarray) -> np.ndarray:
        """
        Score new data: distance to nearest centroid, then mean filter.

        Args:
            scores: 1D array of raw anomaly scores.

        Returns:
            Filtered distance-to-centroid scores.
        """
        X = scores.reshape(-1, 1)
        centroids = self.kmeans.cluster_centers_  # (K, 1)
        # Distance to nearest centroid
        dists = np.min(np.abs(X - centroids.T), axis=1)
        if self.filter_window > 0:
            dists = mean_filter(dists, self.filter_window)
        return dists
