"""TensorFlow-based models for kitsune-zd.

Separated from the main PyTorch/NumPy codebase to keep TensorFlow
as an optional dependency. Install with: pip install -r tf/requirements.txt
"""
try:
    from .lstm_ae import LSTMAutoencoder, LSTMModel
except ImportError:
    LSTMAutoencoder = None
    LSTMModel = None
