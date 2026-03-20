"""
CentroidDetector: K-means++ centroid distance anomaly detector (KMD).

Implements the K-Means Distance (KMD) detector from Section IV-D of the paper:
    Training: K-means++ on the Detection Frame to compute K centroids in
        the 1D anomaly-score space.
    Execution: For each new score, compute distance to the nearest centroid
        as the anomaly measure D(x,t).
    Post-processing: A sliding mean filter smooths the distance signal.

The intuition is that normal traffic scores cluster tightly around learned
centroids, while attack traffic produces scores far from any centroid.
"""
import numpy as np
from sklearn.cluster import KMeans
from .filters import mean_filter


class CentroidDetector:
    """K-Means Distance (KMD) anomaly detector from Section IV-D of the paper.

    Learns a set of K centroids in the 1D anomaly-score space during training,
    then scores new samples by their distance to the nearest centroid. A mean
    filter is applied to the distance signal for temporal smoothing.

    Attributes:
        n_clusters: Number of centroids K for K-means++.
        filter_window: Window size for post-hoc mean filter smoothing.
        kmeans: Fitted sklearn KMeans model (None before training).
    """

    def __init__(self, n_clusters: int = 8, filter_window: int = 100):
        """Initialize the centroid-based detector.

        Args:
            n_clusters: Number of centroids K for K-means++ (int).
                More centroids capture finer structure in the normal-score
                distribution but risk overfitting.
            filter_window: Window size for the sliding mean filter (int).
                Set to 0 to disable smoothing.
        """
        self.n_clusters = n_clusters
        self.filter_window = filter_window
        self.kmeans = None

    def train(self, scores: np.ndarray):
        """Fit K-means++ centroids on the Detection Frame anomaly scores.

        The Detection Frame is a subset of benign traffic scores used to
        learn the normal-score distribution. The number of clusters is
        clamped to min(n_clusters, len(scores)) to handle small datasets.

        Args:
            scores: 1D array of anomaly scores from the Detection Frame
                (training/calibration phase). Shape (N,).
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
        """Compute distance-to-nearest-centroid for new anomaly scores.

        For each score, computes the absolute distance to the nearest of the
        K learned centroids, then applies a sliding mean filter for temporal
        smoothing. Higher distances indicate anomalous behavior.

        Args:
            scores: 1D array of raw anomaly scores from the execution phase.
                Shape (N,).

        Returns:
            np.ndarray: Filtered distance-to-centroid scores of shape (N,).
                Values are non-negative; higher values indicate greater
                anomaly.
        """
        X = scores.reshape(-1, 1)
        centroids = self.kmeans.cluster_centers_  # (K, 1)
        # Distance to nearest centroid
        dists = np.min(np.abs(X - centroids.T), axis=1)
        if self.filter_window > 0:
            dists = mean_filter(dists, self.filter_window)
        return dists
