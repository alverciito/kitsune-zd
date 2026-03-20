"""
DistributionDetector: Harmonic-mean based anomaly detector.

From IEEE paper Eq. 4:
  Training: compute mu_T, sigma_T over Detection Frame.
  Execution: sliding window (W=10000) to compute mu_d, sigma_d.
  Score: D(x,t) = 2 * (mu_T - mu_d)^2 * (sigma_T - sigma_d)^2
                   / ((mu_T - mu_d)^2 + (sigma_T - sigma_d)^2 + eps)

Paired with median filter for smoothing.
"""
import numpy as np
from .filters import median_filter
from ..config import EPSILON


class DistributionDetector:
    """Mean-Variance Distance (MVD) anomaly detector from Section IV-D / Eq. 4.

    Detects anomalies by comparing the local (sliding-window) score
    distribution to the global training distribution using a harmonic-mean
    distance measure. The score captures both mean shift and variance change,
    making it robust to different attack profiles.

    Attributes:
        window_size: Size of the sliding window W for computing local
            statistics (mu_d, sigma_d).
        filter_window: Window size for post-hoc median filter smoothing.
        mu_T: Mean of the training-phase anomaly scores (set by train()).
        sigma_T: Std of the training-phase anomaly scores (set by train()).
    """

    def __init__(self, window_size: int = 10_000, filter_window: int = 100):
        """Initialize the distribution-based detector.

        Args:
            window_size: Sliding window size W for computing local statistics
                (int). Default 10,000 packets per the paper.
            filter_window: Window size for the sliding median filter (int).
                Set to 0 to disable smoothing.
        """
        self.window_size = window_size
        self.filter_window = filter_window
        self.mu_T = None
        self.sigma_T = None

    def train(self, scores: np.ndarray):
        """Compute global training distribution statistics (mu_T, sigma_T).

        These reference statistics are used during execution to measure how
        much the local score distribution deviates from normal behavior.

        Args:
            scores: 1D array of anomaly scores from the Detection Frame
                (training/calibration phase). Shape (N,).
        """
        self.mu_T = np.mean(scores)
        self.sigma_T = np.std(scores)

    def execute(self, scores: np.ndarray) -> np.ndarray:
        """Score new data using the harmonic-mean deviation measure (Eq. 4).

        For each position i, computes local statistics (mu_d, sigma_d) over
        the sliding window [i-W+1, i], then applies the harmonic-mean
        distance formula:
            D(x,t) = 2 * (mu_T - mu_d)^2 * (sigma_T - sigma_d)^2
                     / ((mu_T - mu_d)^2 + (sigma_T - sigma_d)^2 + eps)

        A median filter is then applied for temporal smoothing.

        Args:
            scores: 1D array of raw anomaly scores from the execution phase.
                Shape (N,).

        Returns:
            np.ndarray: Filtered harmonic-mean deviation scores of shape (N,).
                Higher values indicate greater distributional shift from
                training behavior.
        """
        n = len(scores)
        result = np.zeros(n)
        W = self.window_size

        for i in range(n):
            lo = max(0, i - W + 1)
            window = scores[lo:i + 1]
            mu_d = np.mean(window)
            sigma_d = np.std(window)

            delta_mu_sq = (self.mu_T - mu_d) ** 2
            delta_sigma_sq = (self.sigma_T - sigma_d) ** 2
            denom = delta_mu_sq + delta_sigma_sq + EPSILON
            result[i] = 2.0 * delta_mu_sq * delta_sigma_sq / denom

        if self.filter_window > 0:
            result = median_filter(result, self.filter_window)
        return result
