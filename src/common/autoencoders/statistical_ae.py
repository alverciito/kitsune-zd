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
    """Statistical anomaly detector based on z-score deviation from training statistics.

    Implements Eq. 3 from the paper: s(x) = sum(|mu - x_norm| / (sigma + epsilon)).
    Computes per-feature mean and standard deviation during training, then scores
    new samples by summing the absolute z-score deviations across all features.

    Matches the ELMAutoencoder interface (train/execute) so it can be used as a
    drop-in replacement in the KitNET ensemble or output layer. This is the
    default output layer ('stat') per Table II of the paper.

    Attributes:
        n_visible: Number of input features.
        mean: Per-feature mean computed from normalized training data, shape (n_visible,).
        std: Per-feature std computed from normalized training data, shape (n_visible,).
        norm_max: Per-feature maximum for min-max normalization, shape (n_visible,).
        norm_min: Per-feature minimum for min-max normalization, shape (n_visible,).
    """

    def __init__(self, n_visible: int, **kwargs):
        """Initialize the statistical anomaly detector.

        Args:
            n_visible: Number of input features (int).
            **kwargs: Ignored. Accepted for interface compatibility with
                ELMAutoencoder (e.g., hidden_ratio, lr, seed).
        """
        self.n_visible = n_visible
        self.mean = np.zeros(n_visible)
        self.std = np.zeros(n_visible)
        self.norm_max = np.zeros(n_visible)
        self.norm_min = np.zeros(n_visible)

    def _normalize(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Apply min-max normalization to the [0, 1] range.

        During training, computes and stores the per-feature min/max bounds.
        During execution, reuses the bounds learned at training time.

        Args:
            x: Input array of shape (N, n_visible).
            is_training: If True, compute and store normalization bounds
                from this data. If False, use previously stored bounds.

        Returns:
            np.ndarray: Normalized array of the same shape, values in [0, 1].
        """
        if is_training:
            self.norm_max = np.max(x, axis=0)
            self.norm_min = np.min(x, axis=0)
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def train(self, data: np.ndarray) -> np.ndarray:
        """Train by computing mean and std of normalized training data.

        Normalizes the input data to [0, 1] (storing the min/max bounds),
        then computes per-feature mean and standard deviation. Returns
        anomaly scores for the training data itself (via execute()) so
        callers can monitor baseline score distributions.

        Args:
            data: Training data of shape (N, n_visible) or (n_visible,)
                for a single sample.

        Returns:
            np.ndarray: Per-sample anomaly scores of shape (N,), computed
                using the just-learned statistics.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x_norm = self._normalize(data, is_training=True)
        self.mean = np.mean(x_norm, axis=0)
        self.std = np.std(x_norm, axis=0)
        return self.execute(data)

    def execute(self, data: np.ndarray) -> np.ndarray:
        """Score new data using Eq. 3: s(x) = sum(|x_norm - mu| / (sigma + eps)).

        Normalizes input using stored min/max bounds, then computes the sum
        of per-feature absolute z-score deviations from the training mean.
        Higher scores indicate greater anomaly.

        Args:
            data: Input data of shape (N, n_visible) or (n_visible,) for a
                single sample.

        Returns:
            np.ndarray: Per-sample anomaly scores of shape (N,). Each score
                is the sum of absolute z-score deviations across all features.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x_norm = self._normalize(data)
        error = np.abs(x_norm - self.mean) / (self.std + EPSILON)
        return np.sum(error, axis=1)
