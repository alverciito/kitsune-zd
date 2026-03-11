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
from ..utils import create_windows
from ..config import EPSILON, BATCH_SIZE


def _find_valid_num_heads(embed_dim: int, desired_heads: int) -> int:
    """
    Find the largest number of heads <= desired_heads that divides embed_dim.
    PyTorch MultiheadAttention requires embed_dim % num_heads == 0.
    """
    for h in range(min(desired_heads, embed_dim), 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


class TransformerBlock(nn.Module):
    """
    MHA-based autoencoder block reproducing the original TF MultiHeadBlockEncoder.

    Architecture:
      attention_in(Q=K=V=x) -> dense_0(sigmoid) -> dense_1(relu) -> attention_out
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
        x, _ = self.attn_in(x, x, x)
        x = torch.sigmoid(self.dense_0(x))
        x = F.relu(self.dense_1(x))
        x, _ = self.attn_out(x, x, x)
        return x


class TransformerAutoencoder:
    """
    Wrapper matching the ELMAutoencoder/Conv1DAutoencoder interface.
    Same windowing, normalization, and train/execute protocol.
    """

    def __init__(self, n_visible: int, hidden_ratio: float = 0.75,
                 lr: float = 0.001, seq_len: int = 500, device: str = 'cuda'):
        self.n_visible = n_visible
        self.n_hidden = max(1, int(np.ceil(n_visible * hidden_ratio)))
        self.seq_len = seq_len
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = TransformerBlock(n_visible, self.n_hidden).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.norm_min = None
        self.norm_max = None
        self.back_window = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Train on data of shape (N, n_visible).
        Returns: Per-window RMSE array of shape (N - seq_len + 1,).
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
        """
        Score data of shape (N, n_visible). Returns per-sample RMSE.
        BUG FIX #2: NO shuffle during evaluation.
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
