"""
Conv2D Windowed Denoising Autoencoder (PyTorch).

Treats each (seq_len, n_visible) window as a single-channel 2D "image".
Architecture:
  Conv2d(1, 8, kernel_size=3, padding=1, relu) -> Conv2d(8, 16, 3, padding=1, relu)
  -> AdaptiveAvgPool2d(1) -> Linear(16, n_hidden) -> Linear(n_hidden, seq_len * n_visible)
  -> Reshape to (seq_len, n_visible)

Same windowing, normalization, and train/execute protocol as Conv1D/Transformer.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ..utils import create_windows
from ..config import EPSILON, BATCH_SIZE


class Conv2DModel(nn.Module):
    """Conv2D autoencoder: treats (seq_len, n_visible) as a 1-channel image."""

    def __init__(self, n_visible: int, n_hidden: int, seq_len: int):
        super().__init__()
        self.n_visible = n_visible
        self.seq_len = seq_len

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.enc_pool = nn.AdaptiveAvgPool2d(1)  # -> (batch, 16, 1, 1)
        self.enc_fc = nn.Linear(16, n_hidden)

        # Decoder
        self.dec_fc1 = nn.Linear(n_hidden, 16)
        self.dec_fc2 = nn.Linear(16, seq_len * n_visible)

    def forward(self, x):
        # x: (batch, seq_len, n_visible) -> add channel dim
        b = x.shape[0]
        h = x.unsqueeze(1)                        # (batch, 1, seq_len, n_visible)

        # Encode
        h = F.relu(self.enc_conv1(h))             # (batch, 8, seq_len, n_visible)
        h = F.relu(self.enc_conv2(h))             # (batch, 16, seq_len, n_visible)
        h = self.enc_pool(h).view(b, 16)          # (batch, 16)
        z = F.relu(self.enc_fc(h))                # (batch, n_hidden)

        # Decode
        h = F.relu(self.dec_fc1(z))               # (batch, 16)
        h = self.dec_fc2(h)                        # (batch, seq_len * n_visible)
        out = h.view(b, self.seq_len, self.n_visible)
        return out


class Conv2DAutoencoder:
    """
    Wrapper matching the Conv1D/Transformer interface.
    Same windowing, normalization, and train/execute protocol.
    """

    def __init__(self, n_visible: int, hidden_ratio: float = 0.75,
                 lr: float = 0.001, seq_len: int = 500, device: str = 'cuda'):
        self.n_visible = n_visible
        self.n_hidden = max(1, int(np.ceil(n_visible * hidden_ratio)))
        self.seq_len = seq_len
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = Conv2DModel(n_visible, self.n_hidden, seq_len).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.norm_min = None
        self.norm_max = None
        self.back_window = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def train(self, data: np.ndarray) -> np.ndarray:
        """Train on (N, n_visible). Returns per-window RMSE of shape (N - seq_len + 1,)."""
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
        """Score data of shape (N, n_visible). Returns per-sample RMSE."""
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
