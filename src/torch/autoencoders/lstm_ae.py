"""
LSTM Windowed Denoising Autoencoder (PyTorch).

Architecture:
  Encoder: LSTM(n_visible, n_hidden, batch_first=True)
  Decoder: Linear(n_hidden, n_hidden, relu) -> Linear(n_hidden, n_visible, sigmoid)

Input shape: (batch, seq_len, n_visible).
Output: last frame prediction.
Supports autoregressive mode (ar=True).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ...common.utils import create_windows
from ...common.config import EPSILON, BATCH_SIZE


class _LSTMModel(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.encoder = nn.LSTM(n_visible, n_hidden, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_visible),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_visible)
        _, (h_n, _) = self.encoder(x)  # h_n: (1, batch, n_hidden)
        h = h_n.squeeze(0)  # (batch, n_hidden)
        return self.decoder(h)  # (batch, n_visible)


class LSTMAutoencoder:
    """LSTM windowed autoencoder with train/execute API matching other AEs."""

    def __init__(self, n_visible, hidden_ratio=0.75, lr=0.001,
                 seq_len=500, ar=False, device='cuda'):
        self.n_visible = n_visible
        self.n_hidden = max(1, int(np.ceil(n_visible * hidden_ratio)))
        self.lr = lr
        self.seq_len = seq_len
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ar = ar

        torch.manual_seed(1234)
        self.model = _LSTMModel(n_visible, self.n_hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()

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

        dataset = TensorDataset(
            torch.from_numpy(windows),
            torch.from_numpy(targets),
        )

        # Train with shuffle
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            pred = self.model(x_batch)
            loss = self.criterion(pred, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Evaluate without shuffle
        self.model.eval()
        rmse_list = []
        eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE * 4, shuffle=False)
        with torch.no_grad():
            for x_batch, y_batch in eval_loader:
                pred = self.model(x_batch.to(self.device))
                rmse = torch.sqrt(((pred - y_batch.to(self.device)) ** 2).mean(dim=1))
                rmse_list.append(rmse.cpu().numpy())

        # Save back window for execution continuity
        self.back_window = x_norm[-(self.seq_len - 1):]
        return np.concatenate(rmse_list)

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

        dataset = TensorDataset(
            torch.from_numpy(windows),
            torch.from_numpy(targets),
        )

        self.model.eval()
        rmse_list = []
        loader = DataLoader(dataset, batch_size=BATCH_SIZE * 4, shuffle=False)
        with torch.no_grad():
            for x_batch, y_batch in loader:
                pred = self.model(x_batch.to(self.device))
                rmse = torch.sqrt(((pred - y_batch.to(self.device)) ** 2).mean(dim=1))
                rmse_list.append(rmse.cpu().numpy())

        rmse = np.concatenate(rmse_list)
        self.back_window = x_norm[-(self.seq_len - 1):]

        if len(rmse) >= len(data):
            return rmse[-len(data):]
        return rmse
