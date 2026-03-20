"""
Framework-agnostic autoencoders (pure NumPy) for the KitNET ensemble.

Provides two autoencoder types used in the KitNET pipeline (Section III-A):

Exports:
    ELMAutoencoder: Single-hidden-layer denoising autoencoder with online SGD
        (original Kitsune architecture). Used as both ensemble-layer and
        output-layer autoencoders.
    StatisticalAnomaly: Mean/std-based anomaly scorer implementing Eq. 3.
        Drop-in replacement for ELMAutoencoder with the same train/execute
        interface. Default output layer per Table II (DEFAULT_OUTPUT_AE='stat').
"""
from .elm import ELMAutoencoder
from .statistical_ae import StatisticalAnomaly
