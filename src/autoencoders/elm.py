"""
ELM (Extreme Learning Machine style) Denoising Autoencoder.

This is the original Kitsune (NDSS'18) autoencoder: a single-hidden-layer
MLP trained with online SGD in pure numpy. No deep learning framework needed.

Architecture: input -> sigmoid(Wx + b_h) -> sigmoid(W'h + b_v) -> output
Loss: reconstruction RMSE with manual gradient computation.

Also used as the OUTPUT LAYER for all KitNET variants (including Conv1D and
Transformer ensembles), since the output layer aggregates scalar RMSE scores.
"""
import numpy as np
from ..utils import sigmoid
from ..config import EPSILON


class ELMAutoencoder:
    """
    Pure numpy denoising autoencoder with online SGD.

    Supports two modes:
    - Online training: process samples one-by-one, updating norms incrementally.
    - Batch execution: normalize and reconstruct a batch, returning per-sample RMSE.
    """

    def __init__(self, n_visible: int, hidden_ratio: float = 0.75,
                 lr: float = 0.1, corruption_level: float = 0.0, seed: int = 1234):
        self.n_visible = n_visible
        self.n_hidden = max(1, int(np.ceil(n_visible * hidden_ratio)))
        self.lr = lr
        self.corruption_level = corruption_level

        # Online normalization bounds
        self.norm_max = np.full(n_visible, -np.inf)
        self.norm_min = np.full(n_visible, np.inf)

        # Xavier-uniform initialization
        rng = np.random.RandomState(seed)
        a = 1.0 / n_visible
        self.W = rng.uniform(-a, a, (n_visible, self.n_hidden))
        self.h_bias = np.zeros(self.n_hidden)
        self.v_bias = np.zeros(self.n_visible)
        self.W_prime = self.W.T.copy()
        self.rng = rng
        self.n_trained = 0

    def _encode(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(np.dot(x, self.W) + self.h_bias)

    def _decode(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(np.dot(h, self.W_prime) + self.v_bias)

    def _reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self._decode(self._encode(x))

    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Train on data of shape (N, n_visible) using online SGD.
        Updates normalization bounds incrementally per sample.

        Returns:
            Per-sample RMSE array of shape (N,).
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        rmse_list = np.zeros(len(data))

        for i, x in enumerate(data):
            self.n_trained += 1
            # Update norms online
            self.norm_max = np.maximum(self.norm_max, x)
            self.norm_min = np.minimum(self.norm_min, x)

            # Normalize to [0, 1]
            x_norm = (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

            # Optional corruption (dropout)
            if self.corruption_level > 0.0:
                mask = self.rng.binomial(1, 1.0 - self.corruption_level, x_norm.shape)
                tilde_x = x_norm * mask
            else:
                tilde_x = x_norm

            # Forward pass
            y = self._encode(tilde_x)
            z = self._decode(y)

            # Compute gradients
            L_h2 = x_norm - z
            L_h1 = np.dot(L_h2, self.W) * y * (1 - y)
            L_W = np.outer(tilde_x, L_h1) + np.outer(L_h2, y)

            # Update weights
            self.W += self.lr * L_W
            self.h_bias += self.lr * L_h1
            self.v_bias += self.lr * L_h2

            rmse_list[i] = np.sqrt(np.mean(L_h2 ** 2))

        return rmse_list

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Score data of shape (N, n_visible). Returns per-sample RMSE of shape (N,).
        Uses learned normalization bounds (frozen after training).
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x_norm = (data - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)
        z = self._reconstruct(x_norm)
        return np.sqrt(np.mean((x_norm - z) ** 2, axis=1))
