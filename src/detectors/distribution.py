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
    """
    Detect anomalies by comparing local score distribution to training distribution.

    Args:
        window_size: Sliding window size for computing local statistics.
        filter_window: Window size for median filter smoothing (0 = no filter).
    """

    def __init__(self, window_size: int = 10_000, filter_window: int = 100):
        self.window_size = window_size
        self.filter_window = filter_window
        self.mu_T = None
        self.sigma_T = None

    def train(self, scores: np.ndarray):
        """
        Compute training distribution statistics.

        Args:
            scores: 1D array of anomaly scores from training/detection frame.
        """
        self.mu_T = np.mean(scores)
        self.sigma_T = np.std(scores)

    def execute(self, scores: np.ndarray) -> np.ndarray:
        """
        Score new data using Eq. 4 harmonic-mean deviation.

        Args:
            scores: 1D array of raw anomaly scores.

        Returns:
            Filtered distribution deviation scores.
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
