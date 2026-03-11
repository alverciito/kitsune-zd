"""Shared utilities for kitsune-zd."""
import numpy as np
from .config import EPSILON


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def normalize_minmax(x: np.ndarray, norm_min: np.ndarray, norm_max: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] using precomputed bounds."""
    return (x - norm_min) / (norm_max - norm_min + EPSILON)


def create_windows(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create overlapping sliding windows from a 2D array using stride tricks.
    NO SHUFFLE. Fixes Bug #2 from the original code.

    Args:
        x: Input array of shape (N, D).
        window_size: Size of each window.

    Returns:
        Array of shape (N - window_size + 1, window_size, D).
    """
    N, D = x.shape
    if N < window_size:
        raise ValueError(f"Input length {N} < window_size {window_size}")
    n_windows = N - window_size + 1
    strides = (x.strides[0], x.strides[0], x.strides[1])
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_windows, window_size, D), strides=strides
    )


def compute_rmse_per_sample(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-sample RMSE between input and reconstruction.

    Args:
        x, z: Arrays of the same shape (..., D).

    Returns:
        RMSE array of shape (...) with the last axis reduced.
    """
    return np.sqrt(np.mean((x - z) ** 2, axis=-1))
