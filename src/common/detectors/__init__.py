"""
Detector modules for post-processing KitNET anomaly scores into detection decisions.

Provides two anomaly detectors from Section IV-D of the paper, along with
the sliding-window filters they rely on:

Exports:
    CentroidDetector: K-Means Distance (KMD) detector -- scores anomalies by
        distance to nearest learned centroid, smoothed with a mean filter.
    DistributionDetector: Mean-Variance Distance (MVD) detector -- scores
        anomalies by harmonic-mean deviation of local vs. training statistics,
        smoothed with a median filter.
    mean_filter: Sliding-window moving average for score smoothing.
    median_filter: Sliding-window median for score smoothing.
"""
from .centroid import CentroidDetector
from .distribution import DistributionDetector
from .filters import mean_filter, median_filter
