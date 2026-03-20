"""
Shared utilities for kitsune-zd.

Provides helper functions used across the KitNET pipeline: activation functions,
normalization, sliding-window creation for time-series autoencoders, and
per-sample reconstruction error computation.
"""
import numpy as np
from .config import EPSILON


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the numerically stable sigmoid activation function.

    Clips input to [-500, 500] before exponentiation to avoid floating-point
    overflow warnings. Used by the ELMAutoencoder encoder/decoder layers.

    Args:
        x: Input array of any shape (np.ndarray).

    Returns:
        np.ndarray: Element-wise sigmoid values in (0, 1), same shape as input.
    """
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_minmax(x: np.ndarray, norm_min: np.ndarray, norm_max: np.ndarray) -> np.ndarray:
    """Min-max normalize features to the [0, 1] range using precomputed bounds.

    Applies the transformation: (x - min) / (max - min + epsilon).
    EPSILON is added to the denominator to prevent division by zero for
    constant features.

    Args:
        x: Input array of shape (N, D) or (D,).
        norm_min: Per-feature minimum values of shape (D,).
        norm_max: Per-feature maximum values of shape (D,).

    Returns:
        np.ndarray: Normalized array of the same shape as x, with values in [0, 1].
    """
    return (x - norm_min) / (norm_max - norm_min + EPSILON)


def create_windows(x: np.ndarray, window_size: int) -> np.ndarray:
    """Create overlapping sliding windows from a 2D array using stride tricks.

    Implements the windowing strategy from Section III-B of the paper for
    time-series reconstruction (TSR) mode. Windows are created in sequential
    order (NO SHUFFLE) to preserve temporal context. This fixes Bug #2 from
    the original Kitsune codebase, where random shuffling destroyed temporal
    ordering.

    Uses np.lib.stride_tricks.as_strided for zero-copy window creation,
    which is memory-efficient even for large datasets.

    Args:
        x: Input array of shape (N, D) where N is the number of time steps
            and D is the feature dimensionality.
        window_size: Size of each window in packets (default: 800 per Table II).

    Returns:
        np.ndarray: Array of shape (N - window_size + 1, window_size, D)
            containing overlapping windows with stride 1.

    Raises:
        ValueError: If input length N is less than window_size.
    """
    N, D = x.shape
    if N < window_size:
        raise ValueError(f"Input length {N} < window_size {window_size}")
    n_windows = N - window_size + 1
    strides = (x.strides[0], x.strides[0], x.strides[1])
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_windows, window_size, D), strides=strides
    )


def create_windows_ar(x: np.ndarray, window_size: int):
    """Create autoregressive (AR) windows for next-frame prediction.

    Implements the AR windowing mode from Section III-B of the paper. Each
    window is split into an input context (all frames except the last) and a
    target (the last frame). The model learns to predict the next unseen
    packet given a history of window_size - 1 packets.

    Args:
        x: Input array of shape (N, D) where N is the number of time steps
            and D is the feature dimensionality.
        window_size: Total window size including the target frame. The input
            context will have (window_size - 1) frames.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair (inputs, targets) where:
            - inputs: shape (N - window_size + 1, window_size - 1, D)
            - targets: shape (N - window_size + 1, D)

    Raises:
        ValueError: If input length N is less than window_size (propagated
            from create_windows).
    """
    windows = create_windows(x, window_size)
    inputs = windows[:, :-1, :]
    targets = windows[:, -1, :]
    return inputs, targets


def compute_rmse_per_sample(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute per-sample RMSE between input and its reconstruction.

    This is the core anomaly scoring function for all autoencoders in
    the KitNET ensemble. Higher RMSE indicates a sample that the
    autoencoder fails to reconstruct well, signaling a potential anomaly.

    Args:
        x: Original input array of shape (..., D).
        z: Reconstructed array of the same shape as x.

    Returns:
        np.ndarray: RMSE array of shape (...) with the last (feature) axis
            reduced via mean-squared-error then square root.
    """
    return np.sqrt(np.mean((x - z) ** 2, axis=-1))
