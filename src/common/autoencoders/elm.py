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
    """Pure-numpy denoising autoencoder with online SGD, based on Kitsune (NDSS'18).

    Implements a single-hidden-layer autoencoder with tied weights, sigmoid
    activations, and optional input corruption (denoising). Training uses
    sample-by-sample online SGD with incremental min-max normalization.

    Architecture (Section III-A of the paper):
        input -> sigmoid(W * x + b_h) -> sigmoid(W' * h + b_v) -> output

    This class serves dual purposes in the KitNET pipeline:
    1. As an ensemble-layer autoencoder that reconstructs a subset of features.
    2. As the output-layer autoencoder that aggregates per-AE RMSE scores.

    Bug Fix #4: The decoder weight matrix W' is kept in sync with the
    encoder W after each gradient update (W' = W^T).

    Attributes:
        n_visible: Number of input/output neurons.
        n_hidden: Number of hidden neurons, computed as ceil(n_visible * hidden_ratio).
        lr: Learning rate for SGD updates.
        corruption_level: Fraction of inputs zeroed out for denoising (0.0 = none).
        norm_max: Per-feature maximum values learned during training.
        norm_min: Per-feature minimum values learned during training.
        W: Encoder weight matrix of shape (n_visible, n_hidden).
        W_prime: Decoder weight matrix of shape (n_hidden, n_visible).
        h_bias: Hidden layer bias of shape (n_hidden,).
        v_bias: Visible layer bias of shape (n_visible,).
        n_trained: Count of samples processed during training.
    """

    def __init__(self, n_visible: int, hidden_ratio: float = 0.22,
                 lr: float = 0.001, corruption_level: float = 0.0, seed: int = 1234):
        """Initialize the ELM autoencoder with Xavier-uniform weights.

        Args:
            n_visible: Number of input features (visible units). For ensemble
                AEs this equals the cluster size; for the output AE this
                equals the number of ensemble AEs.
            hidden_ratio: Ratio of hidden units to visible units (float).
                Default 0.22 per Table II (78% compression).
            lr: Learning rate for online SGD (float). Default 0.001 per Table II.
            corruption_level: Fraction of inputs to zero out during training
                for denoising (float in [0, 1]). Default 0.0 (no corruption).
            seed: Random seed for weight initialization and corruption mask
                generation (int).
        """
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
        """Encode input to hidden representation: h = sigmoid(W * x + b_h).

        Args:
            x: Input vector of shape (n_visible,) or batch (N, n_visible).

        Returns:
            np.ndarray: Hidden representation of shape (n_hidden,) or (N, n_hidden).
        """
        return sigmoid(np.dot(x, self.W) + self.h_bias)

    def _decode(self, h: np.ndarray) -> np.ndarray:
        """Decode hidden representation to reconstruction: z = sigmoid(W' * h + b_v).

        Args:
            h: Hidden vector of shape (n_hidden,) or batch (N, n_hidden).

        Returns:
            np.ndarray: Reconstructed input of shape (n_visible,) or (N, n_visible).
        """
        return sigmoid(np.dot(h, self.W_prime) + self.v_bias)

    def _reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass: encode then decode.

        Args:
            x: Input vector of shape (n_visible,) or batch (N, n_visible).

        Returns:
            np.ndarray: Reconstruction of the same shape as x.
        """
        return self._decode(self._encode(x))

    def train(self, data: np.ndarray) -> np.ndarray:
        """Train on data using sample-by-sample online SGD.

        For each sample: (1) updates incremental min-max normalization bounds,
        (2) normalizes to [0,1], (3) optionally applies corruption mask,
        (4) performs forward pass, (5) computes gradients via backpropagation,
        and (6) updates weights and biases.

        The gradient computation follows the tied-weight denoising autoencoder
        formulation from the original Kitsune (NDSS'18) paper.

        Args:
            data: Training data of shape (N, n_visible) or (n_visible,) for
                a single sample. Each row is one packet's feature vector.

        Returns:
            np.ndarray: Per-sample RMSE array of shape (N,), measuring
                reconstruction error during training. Useful for monitoring
                convergence.
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
            self.W_prime = self.W.T.copy()  # Bug #4 fix: keep decoder in sync
            self.h_bias += self.lr * L_h1
            self.v_bias += self.lr * L_h2

            rmse_list[i] = np.sqrt(np.mean(L_h2 ** 2))

        return rmse_list

    def execute(self, data: np.ndarray) -> np.ndarray:
        """Score data by computing reconstruction RMSE (inference/execution phase).

        Normalizes input using the min-max bounds learned during training
        (frozen), reconstructs via the encoder-decoder, and returns the
        per-sample RMSE as the anomaly score. Higher scores indicate greater
        deviation from learned normal behavior.

        Args:
            data: Input data of shape (N, n_visible) or (n_visible,) for a
                single sample.

        Returns:
            np.ndarray: Per-sample RMSE anomaly scores of shape (N,).
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x_norm = (data - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)
        z = self._reconstruct(x_norm)
        return np.sqrt(np.mean((x_norm - z) ** 2, axis=1))
