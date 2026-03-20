"""
Deep MLP (Multilayer) Windowed Denoising Autoencoder (TensorFlow/Keras).
Ported from tf_original/src/models/networks/multilayer.py

Architecture (from tf_original, adapted for windowed input):
  Encoder:
    Flatten(seq_len * n_visible)
    Dense(input_dim // 4, relu)
    Dense(n_hidden, relu)
  Decoder:
    Dense(input_dim // 4, relu)
    Dense(n_visible, sigmoid)

Unlike the single-layer ELM, this uses a multi-layer encoder/decoder
with batch training on windowed sequences.

Input shape: (batch, seq_len, n_visible).
Output: last frame prediction (batch, n_visible).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from ...common.utils import create_windows
from ...common.config import EPSILON, BATCH_SIZE


class _DeepMLPModel(tf.keras.Model):
    """Deep MLP autoencoder adapted from tf_original multilayer architecture."""

    def __init__(self, n_visible: int, n_hidden: int, seq_len: int):
        super().__init__()
        input_dim = seq_len * n_visible
        h1 = max(1, input_dim // 4)

        self.flatten = tf.keras.layers.Flatten()

        # Encoder: flatten -> h1 -> n_hidden
        self.enc1 = tf.keras.layers.Dense(h1, activation='relu')
        self.enc2 = tf.keras.layers.Dense(n_hidden, activation='relu')

        # Decoder: n_hidden -> h1 -> n_visible
        self.dec1 = tf.keras.layers.Dense(h1, activation='relu')
        self.dec2 = tf.keras.layers.Dense(n_visible, activation='sigmoid')

    def call(self, x):
        # x: (batch, seq_len, n_visible) -> flatten
        h = self.flatten(x)

        # Encode
        h = self.enc1(h)
        h = self.enc2(h)

        # Decode
        h = self.dec1(h)
        h = self.dec2(h)
        return h


class DeepMLPAutoencoder:
    """
    Wrapper matching the PyTorch DeepMLPAutoencoder interface.

    Uses TensorFlow internally. Handles normalization, windowing,
    training and evaluation with the same API as PyTorch variants.
    """

    def __init__(self, n_visible: int, hidden_ratio: float = 0.75,
                 lr: float = 0.001, seq_len: int = 500, ar: bool = False,
                 device: str = 'cuda', **kwargs):
        self.n_visible = n_visible
        self.n_hidden = max(1, int(np.ceil(n_visible * hidden_ratio)))
        self.seq_len = seq_len
        self.ar = ar

        tf.random.set_seed(1234)
        self.model = _DeepMLPModel(n_visible, self.n_hidden, seq_len)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        self.norm_min = None
        self.norm_max = None
        self.back_window = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def _make_windows_and_targets(self, x_norm: np.ndarray):
        """
        Create (window, target) pairs.
        - TSR mode (ar=False): window = x[i:i+seq_len], target = x[i+seq_len-1] (last frame)
        - AR mode (ar=True): window = x[i:i+seq_len], target = x[i+seq_len] (next frame)
        """
        if self.ar:
            n_samples = len(x_norm) - self.seq_len
            if n_samples <= 0:
                return np.empty((0, self.seq_len, self.n_visible)), np.empty((0, self.n_visible))
            windows = create_windows(x_norm[:len(x_norm) - 1], self.seq_len).copy()
            targets = x_norm[self.seq_len:self.seq_len + len(windows)]
            min_len = min(len(windows), len(targets))
            return windows[:min_len], targets[:min_len]
        else:
            windows = create_windows(x_norm, self.seq_len).copy()
            targets = x_norm[self.seq_len - 1:self.seq_len - 1 + len(windows)]
            min_len = min(len(windows), len(targets))
            return windows[:min_len], targets[:min_len]

    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Train on data of shape (N, n_visible).
        Returns: Per-window RMSE array.
        """
        self.norm_max = np.max(data, axis=0)
        self.norm_min = np.min(data, axis=0)
        x_norm = self._normalize(data).astype(np.float32)

        windows, targets = self._make_windows_and_targets(x_norm)
        if len(windows) == 0:
            return np.array([])

        self.model.fit(
            windows, targets,
            batch_size=BATCH_SIZE, epochs=1, verbose=0, shuffle=True
        )

        predictions = self.model.predict(windows, batch_size=BATCH_SIZE * 4, verbose=0)
        rmse = np.sqrt(np.mean((targets - predictions) ** 2, axis=1))

        self.back_window = x_norm[-(self.seq_len - 1):]
        return rmse

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Score data of shape (N, n_visible). Returns per-sample RMSE.
        Prepends saved back_window for sliding window continuity.
        """
        x_norm = self._normalize(data).astype(np.float32)

        if self.back_window is not None:
            x_ext = np.concatenate([self.back_window, x_norm], axis=0)
        else:
            x_ext = x_norm

        if len(x_ext) < self.seq_len:
            return np.zeros(len(data))

        windows, targets = self._make_windows_and_targets(x_ext)
        if len(windows) == 0:
            return np.zeros(len(data))

        predictions = self.model.predict(windows, batch_size=BATCH_SIZE * 4, verbose=0)
        rmse = np.sqrt(np.mean((targets - predictions) ** 2, axis=1))

        self.back_window = x_norm[-(self.seq_len - 1):]

        if len(rmse) >= len(data):
            return rmse[-len(data):]
        return rmse
