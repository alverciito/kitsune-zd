"""
Statistical Anomaly Detector (NumPy).

Ported from the original TensorFlow codebase (stdev.py).
Implements Eq. 3 from the paper: s(x) = |μ - x| / (σ + ε)

No neural network — just mean/std based anomaly scoring.
Used as the "STAT" baseline and optionally as the output layer.
"""
import numpy as np
from ..config import EPSILON


class StatisticalAnomaly:
    """
    Simple anomaly detector based on deviation from training statistics.

    Matches the ELMAutoencoder interface (train/execute) so it can be
    used as a drop-in replacement in the KitNET ensemble or output layer.
    """

    def __init__(self, n_visible: int, **kwargs):
        self.n_visible = n_visible
        self.mean = np.zeros(n_visible)
        self.std = np.zeros(n_visible)
        self.norm_max = np.zeros(n_visible)
        self.norm_min = np.zeros(n_visible)

    def _normalize(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        if is_training:
            self.norm_max = np.max(x, axis=0)
            self.norm_min = np.min(x, axis=0)
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Train: compute mean and std of normalized data.
        Returns per-sample anomaly scores (same as execute).
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x_norm = self._normalize(data, is_training=True)
        self.mean = np.mean(x_norm, axis=0)
        self.std = np.std(x_norm, axis=0)
        return self.execute(data)

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Score: sum of |x_norm - mean| / (std + epsilon) per sample.
        Returns per-sample anomaly score of shape (N,).
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x_norm = self._normalize(data)
        error = np.abs(x_norm - self.mean) / (self.std + EPSILON)
        return np.sum(error, axis=1)
