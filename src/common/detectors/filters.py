"""
Sliding-window mean and median filters for anomaly score smoothing.

Used as post-processing steps by the CentroidDetector (mean filter) and
DistributionDetector (median filter) to reduce noise in the anomaly score
time series before threshold-based detection. See Section IV-D of the paper.
"""
import numpy as np


def mean_filter(scores: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a sliding-window mean (moving average) filter to anomaly scores.

    Uses numpy convolution with a uniform kernel for efficient computation.
    The output is centered so that the smoothed signal aligns temporally
    with the input. Edge effects are handled by the 'full' convolution
    mode with offset-based truncation.

    Used by CentroidDetector (KMD) for post-hoc score smoothing.

    Args:
        scores: 1D array of anomaly scores (np.ndarray). Shape (N,).
        window_size: Size of the sliding window (int). If <= 1, returns
            a copy of the input unchanged.

    Returns:
        np.ndarray: Smoothed scores of the same length (N,).
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
    """Apply a sliding-window median filter to anomaly scores.

    Computes the median over a symmetric window centered on each position.
    At the edges, the window is truncated (partial windows). The median
    filter is more robust to outlier spikes than the mean filter.

    Used by DistributionDetector (MVD) for post-hoc score smoothing.

    Args:
        scores: 1D array of anomaly scores (np.ndarray). Shape (N,).
        window_size: Size of the sliding window (int). If <= 1, returns
            a copy of the input unchanged.

    Returns:
        np.ndarray: Smoothed scores of the same length (N,).
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
