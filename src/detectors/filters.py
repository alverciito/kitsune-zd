"""Sliding window mean and median filters for anomaly score smoothing."""
import numpy as np


def mean_filter(scores: np.ndarray, window_size: int) -> np.ndarray:
    """
    Sliding window mean filter.

    Args:
        scores: 1D array of anomaly scores.
        window_size: Size of the sliding window.

    Returns:
        Smoothed scores of the same length (edges use partial windows).
    """
    if window_size <= 1:
        return scores.copy()
    n = len(scores)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode='full')
    # Center the output to match input length
    offset = (len(smoothed) - n) // 2
    return smoothed[offset:offset + n]


def median_filter(scores: np.ndarray, window_size: int) -> np.ndarray:
    """
    Sliding window median filter.

    Args:
        scores: 1D array of anomaly scores.
        window_size: Size of the sliding window.

    Returns:
        Smoothed scores of the same length (edges use partial windows).
    """
    if window_size <= 1:
        return scores.copy()
    n = len(scores)
    half = window_size // 2
    result = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.median(scores[lo:hi])
    return result
