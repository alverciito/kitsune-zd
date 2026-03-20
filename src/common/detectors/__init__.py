"""Detector modules: post-processing anomaly scores into detection decisions."""
from .centroid import CentroidDetector
from .distribution import DistributionDetector
from .filters import mean_filter, median_filter
