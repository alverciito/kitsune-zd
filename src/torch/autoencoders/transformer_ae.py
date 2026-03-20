"""
Transformer (Multi-Head Attention) Windowed Denoising Autoencoder (PyTorch).

Replaces the original TensorFlow MultiHeadBlockEncoder. Architecture:
  MultiHeadAttention -> Dense(sigmoid) -> Dense(relu) -> MultiHeadAttention

Input/Output shape: (batch, seq_len, n_visible).

BUG FIX #2: Evaluation NEVER shuffles data.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ...common.utils import create_windows
from ...common.config import EPSILON, BATCH_SIZE


def _find_valid_num_heads(embed_dim: int, desired_heads: int) -> int:
    """Find the largest valid number of attention heads.

    PyTorch's MultiheadAttention requires ``embed_dim % num_heads == 0``.
    This function finds the largest number of heads at or below
    ``desired_heads`` that satisfies this divisibility constraint.

    Args:
        embed_dim: Embedding dimension (must be positive).
        desired_heads: Preferred number of attention heads.

    Returns:
        int: Largest valid head count <= desired_heads that evenly
            divides embed_dim. Returns 1 as a fallback.
    """
    for h in range(min(desired_heads, embed_dim), 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


class TransformerBlock(nn.Module):
    """Multi-Head Attention autoencoder block for windowed anomaly detection.

    Reproduces the original TensorFlow MultiHeadBlockEncoder architecture
    (Section III-C of the paper) using PyTorch's MultiheadAttention.

    Architecture:
        attention_in(Q=K=V=x) - self-attention over the input sequence
        -> Linear(n_visible, n_hidden) + sigmoid - bottleneck compression
        -> Linear(n_hidden, n_visible) + ReLU - feature reconstruction
        -> attention_out(Q=K=V=x) - self-attention over reconstructed sequence

    The number of attention heads is automatically adjusted to the largest
    value <= n_hidden that evenly divides n_visible (PyTorch requirement).

    Input shape:
        ``(batch, seq_len, n_visible)`` - batch of sliding windows.
    Output shape:
        ``(batch, seq_len, n_visible)`` - reconstructed windows.

    Args:
        n_visible: Number of input features per timestep (also the
            embed_dim for multi-head attention).
        n_hidden: Bottleneck dimensionality and desired number of
            attention heads (adjusted for divisibility).
    """

    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        num_heads = _find_valid_num_heads(n_visible, n_hidden)
        self.attn_in = nn.MultiheadAttention(
            embed_dim=n_visible, num_heads=num_heads, batch_first=True
        )
        self.dense_0 = nn.Linear(n_visible, n_hidden)
        self.dense_1 = nn.Linear(n_hidden, n_visible)
        self.attn_out = nn.MultiheadAttention(
            embed_dim=n_visible, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        """Forward pass through the attention-bottleneck-attention pipeline.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_visible)``.

        Returns:
            torch.Tensor: Reconstructed tensor of shape
                ``(batch, seq_len, n_visible)``.
        """
        x, _ = self.attn_in(x, x, x)
        x = torch.sigmoid(self.dense_0(x))
        x = F.relu(self.dense_1(x))
        x, _ = self.attn_out(x, x, x)
        return x


class TransformerAutoencoder:
    """Windowed Transformer autoencoder with online normalization.

    Wraps TransformerBlock with sliding-window preprocessing, incremental
    min-max normalization, and back_window continuity for streaming
    execution across chunks. Uses temporal sequence reconstruction (TSR)
    windowing where each window is reconstructed in full via self-attention.

    Follows the same train/execute protocol as Conv1DAutoencoder and
    Conv2DAutoencoder: train with shuffled DataLoader, evaluate without
    shuffle, prepend back_window during execution for continuity.

    Args:
        n_visible: Number of input features per packet.
        hidden_ratio: Compression ratio for bottleneck size (default: 0.75).
        lr: Learning rate for Adam optimizer (default: 0.001).
        seq_len: Sliding window length in packets (default: 500).
        device: PyTorch device string ('cuda' or 'cpu').
    """

    def __init__(self, n_visible: int, hidden_ratio: float = 0.75,
                 lr: float = 0.001, seq_len: int = 500, device: str = 'cuda'):
        self.n_visible = n_visible
        self.n_hidden = max(1, int(np.ceil(n_visible * hidden_ratio)))
        self.seq_len = seq_len
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(1234)
        self.model = TransformerBlock(n_visible, self.n_hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()

        self.norm_min = None
        self.norm_max = None
        self.back_window = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to scale features to [0, 1].

        Args:
            x: Input array of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Normalized array of same shape, values in [0, 1].
        """
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def train(self, data: np.ndarray) -> np.ndarray:
        """Train the Transformer model on a batch of packet features.

        Computes per-feature min/max, normalizes, creates sliding windows,
        trains for 1 epoch with shuffled DataLoader (Adam, MSE loss), then
        evaluates without shuffle. Saves back_window for execution continuity.

        Args:
            data: Training data of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Per-window RMSE array of shape ``(N - seq_len + 1,)``.
        """
        self.norm_max = np.max(data, axis=0)
        self.norm_min = np.min(data, axis=0)
        x_norm = self._normalize(data).astype(np.float32)

        windows = create_windows(x_norm, self.seq_len).copy()
        tensor_x = torch.from_numpy(windows).to(self.device)

        # Train with shuffle
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.model.train()
        for x_batch, y_batch in loader:
            self.optimizer.zero_grad()
            z_batch = self.model(x_batch)
            loss = self.criterion(z_batch, y_batch)
            loss.backward()
            self.optimizer.step()

        # Evaluate without shuffle
        self.model.eval()
        rmse_list = []
        eval_loader = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE * 4, shuffle=False)
        with torch.no_grad():
            for (xb,) in eval_loader:
                zb = self.model(xb)
                mse = torch.mean((xb - zb) ** 2, dim=(1, 2))
                rmse_list.append(torch.sqrt(mse).cpu().numpy())

        self.back_window = x_norm[-(self.seq_len - 1):]
        return np.concatenate(rmse_list)

    def execute(self, data: np.ndarray) -> np.ndarray:
        """Score new data using the trained Transformer model.

        Normalizes input, prepends the saved back_window for sliding-window
        continuity, computes per-window RMSE without shuffle, and trims
        output to match input length. Updates back_window for subsequent calls.

        Bug Fix #2: The evaluation DataLoader never shuffles.

        Args:
            data: Input data of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Per-sample RMSE scores of shape ``(N,)``.
                Returns zeros if extended data is shorter than seq_len.
        """
        x_norm = self._normalize(data).astype(np.float32)

        if self.back_window is not None:
            x_ext = np.concatenate([self.back_window, x_norm], axis=0)
        else:
            x_ext = x_norm

        if len(x_ext) < self.seq_len:
            return np.zeros(len(data))

        windows = create_windows(x_ext, self.seq_len).copy()
        tensor_x = torch.from_numpy(windows).to(self.device)

        self.model.eval()
        rmse_list = []
        loader = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE * 4, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                zb = self.model(xb)
                mse = torch.mean((xb - zb) ** 2, dim=(1, 2))
                rmse_list.append(torch.sqrt(mse).cpu().numpy())

        rmse = np.concatenate(rmse_list)
        self.back_window = x_norm[-(self.seq_len - 1):]

        if len(rmse) >= len(data):
            return rmse[-len(data):]
        return rmse
