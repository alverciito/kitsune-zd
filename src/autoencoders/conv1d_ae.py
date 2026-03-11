"""
Conv1D Windowed Denoising Autoencoder (PyTorch).

Replaces the original TensorFlow implementation. Architecture matches the original:
  Conv1D(in=n_visible, out=3, kernel_size=seq_len, padding='same', relu)
  -> Linear(3, n_hidden, relu)
  -> Linear(n_hidden, n_visible, relu)

Input/Output shape: (batch, seq_len, n_visible).

BUG FIX #2: Evaluation NEVER shuffles data. Only training DataLoader shuffles.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ..utils import create_windows
from ..config import EPSILON, BATCH_SIZE


class Conv1DModel(nn.Module):
    """Conv1D autoencoder model. Reconstructs windowed input sequences."""

    def __init__(self, n_visible: int, n_hidden: int, seq_len: int):
        super().__init__()
        # Conv1D: PyTorch expects (batch, channels, length)
        # Original TF: Conv1D(filters=3, kernel_size=seq_len, padding='same')
        self.conv = nn.Conv1d(n_visible, 3, kernel_size=seq_len, padding='same')
        self.dense1 = nn.Linear(3, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_visible)

    def forward(self, x):
        # x: (batch, seq_len, n_visible)
        x = x.permute(0, 2, 1)       # -> (batch, n_visible, seq_len)
        x = F.relu(self.conv(x))      # -> (batch, 3, seq_len)
        x = x.permute(0, 2, 1)       # -> (batch, seq_len, 3)
        x = F.relu(self.dense1(x))   # -> (batch, seq_len, n_hidden)
        x = F.relu(self.dense2(x))   # -> (batch, seq_len, n_visible)
        return x


class Conv1DAutoencoder:
    """
    Wrapper matching the ELMAutoencoder interface.

    Handles normalization, windowing, training (with shuffle) and evaluation
    (WITHOUT shuffle - Bug #2 fix).
    """

    def __init__(self, n_visible: int, hidden_ratio: float = 0.75,
                 lr: float = 0.001, seq_len: int = 500, device: str = 'cuda'):
        self.n_visible = n_visible
        self.n_hidden = max(1, int(np.ceil(n_visible * hidden_ratio)))
        self.seq_len = seq_len
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = Conv1DModel(n_visible, self.n_hidden, seq_len).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Normalization (set during training)
        self.norm_min = None
        self.norm_max = None
        # Back window for continuity during execution
        self.back_window = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Train on data of shape (N, n_visible).
        1. Compute min/max norms
        2. Normalize to [0,1]
        3. Create sliding windows
        4. Train model for 1 epoch WITH shuffled DataLoader
        5. Evaluate WITHOUT shuffle -> per-window RMSE

        Returns: Per-window RMSE array of shape (N - seq_len + 1,).
        """
        # Compute and store norms
        self.norm_max = np.max(data, axis=0)
        self.norm_min = np.min(data, axis=0)
        x_norm = self._normalize(data).astype(np.float32)

        # Create windows: (N - seq_len + 1, seq_len, n_visible)
        windows = create_windows(x_norm, self.seq_len).copy()
        tensor_x = torch.from_numpy(windows).to(self.device)

        # Train with shuffled DataLoader
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.model.train()
        for x_batch, y_batch in loader:
            self.optimizer.zero_grad()
            z_batch = self.model(x_batch)
            loss = self.criterion(z_batch, y_batch)
            loss.backward()
            self.optimizer.step()

        # Evaluate WITHOUT shuffle for per-window RMSE
        self.model.eval()
        rmse_list = []
        eval_loader = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE * 4, shuffle=False)
        with torch.no_grad():
            for (xb,) in eval_loader:
                zb = self.model(xb)
                # Per-window MSE: mean over (seq_len, n_visible) -> per-window scalar
                mse = torch.mean((xb - zb) ** 2, dim=(1, 2))
                rmse_list.append(torch.sqrt(mse).cpu().numpy())

        # Save back window for execution continuity
        self.back_window = x_norm[-(self.seq_len - 1):]

        return np.concatenate(rmse_list)

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Score data of shape (N, n_visible). Returns per-sample RMSE.

        Prepends saved back_window for sliding window continuity,
        so output length == N (not N - seq_len + 1).

        BUG FIX #2: NO shuffle during evaluation.
        """
        x_norm = self._normalize(data).astype(np.float32)

        # Prepend back window for continuity
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

        # Save back window for next execution call
        self.back_window = x_norm[-(self.seq_len - 1):]

        # Trim to match input length (remove extra from back_window prepend)
        # x_ext has len(back_window) + len(data) samples
        # windows has len(x_ext) - seq_len + 1 entries
        # We want the last len(data) RMSE values
        if len(rmse) >= len(data):
            return rmse[-len(data):]
        return rmse
