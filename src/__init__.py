"""
KitNET: Ensemble of Autoencoders for Network Intrusion Detection.

Provides two backends: 'torch' (PyTorch) and 'tf' (TensorFlow).
Set KITSUNE_BACKEND env var to select (default: 'torch').

Common (framework-agnostic) components are always available:
  config, database, clustering, detectors, ELM, Statistical AE.
"""
import os

_BACKEND = os.environ.get('KITSUNE_BACKEND', 'torch')

# Re-export common components
from .common.config import *
from .common.database import load_attack, load_cic2017, load_cic2018, load_aci_iot
from .common.clustering import get_clustering
from .common.detector import threshold_sweep, windowdiff, plot_roc, save_results
from .common.detectors import CentroidDetector, DistributionDetector
from .common.autoencoders import ELMAutoencoder, StatisticalAnomaly

# Backend-specific KitNET
if _BACKEND == 'tf':
    from .tf import KitNET
else:
    from .torch import KitNET
