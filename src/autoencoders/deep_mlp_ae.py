"""
Deep MLP (Multilayer) Windowed Denoising Autoencoder (PyTorch).

Unlike the single-layer ELM, this uses a 3-layer encoder/decoder with batch training:
  Encoder: n_visible -> h1 -> h2 (bottleneck)
  Decoder: h2 -> h1 -> n_visible

Same windowing, normalization, and train/execute protocol as Conv1D/Transformer.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ..utils import create_windows
from ..config import EPSILON, BATCH_SIZE


class DeepMLPModel(nn.Module):
    """Multilayer MLP autoencoder operating on windowed sequences."""

    def __init__(self, n_visible: int, n_hidden: int, seq_len: int):
        super().__init__()
        self.n_visible = n_visible
        self.seq_len = seq_len
        input_dim = seq_len * n_visible

        # Intermediate sizes
        h1 = max(1, input_dim // 4)
        h2 = max(1, n_hidden)

        # Encoder: input -> h1 -> h2
        self.enc1 = nn.Linear(input_dim, h1)
        self.enc2 = nn.Linear(h1, h2)

        # Decoder: h2 -> h1 -> input
        self.dec1 = nn.Linear(h2, h1)
        self.dec2 = nn.Linear(h1, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, n_visible) -> flatten
        b = x.shape[0]
        h = x.view(b, -1)

        # Encode
        h = F.relu(self.enc1(h))
        h = F.relu(self.enc2(h))

        # Decode
        h = F.relu(self.dec1(h))
        h = self.dec2(h)

        return h.view(b, self.seq_len, self.n_visible)


class DeepMLPAutoencoder:
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

        self.model = DeepMLPModel(n_visible, self.n_hidden, seq_len).to(self.device)
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
