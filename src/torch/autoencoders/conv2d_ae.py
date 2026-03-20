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
from ...common.utils import create_windows
from ...common.config import EPSILON, BATCH_SIZE


class Conv2DModel(nn.Module):
    """Conv2D encoder-decoder treating packet windows as single-channel images.

    Architecture (Section III-C of the paper):
        Encoder:
            Conv2d(1, 8, kernel_size=3, padding=1) + ReLU
            -> Conv2d(8, 16, kernel_size=3, padding=1) + ReLU
            -> AdaptiveAvgPool2d(1) -> flatten to (batch, 16)
            -> Linear(16, n_hidden) + ReLU
        Decoder:
            Linear(n_hidden, 16) + ReLU
            -> Linear(16, seq_len * n_visible)
            -> reshape to (batch, seq_len, n_visible)

    Each ``(seq_len, n_visible)`` sliding window is treated as a
    single-channel 2D image. The encoder extracts spatial features
    via convolutions, compresses through adaptive pooling, and maps
    to a bottleneck. The decoder reconstructs the full window from
    the bottleneck via fully-connected layers.

    Input shape:
        ``(batch, seq_len, n_visible)`` - batch of sliding windows.
    Output shape:
        ``(batch, seq_len, n_visible)`` - reconstructed windows.

    Args:
        n_visible: Number of input features per timestep (image width).
        n_hidden: Bottleneck dimensionality.
        seq_len: Window length in timesteps (image height).
    """

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
        """Forward pass: encode the 2D window and decode to reconstruct it.

        Adds a channel dimension, applies two convolutional layers with
        adaptive pooling, projects through the bottleneck, and reshapes
        the decoder output back to the original window dimensions.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_visible)``.

        Returns:
            torch.Tensor: Reconstructed tensor of shape
                ``(batch, seq_len, n_visible)``.
        """
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
    """Windowed Conv2D autoencoder with online normalization.

    Wraps Conv2DModel with sliding-window preprocessing, incremental
    min-max normalization, and back_window continuity for streaming
    execution across chunks. Each ``(seq_len, n_visible)`` window is
    treated as a single-channel 2D image for convolution.

    Follows the same windowing, normalization, and train/execute protocol
    as Conv1DAutoencoder and TransformerAutoencoder.

    Training procedure:
        1. Compute per-feature min/max from training data.
        2. Normalize to [0, 1] via min-max scaling.
        3. Create sliding windows of shape ``(N - seq_len + 1, seq_len, n_visible)``.
        4. Train Conv2DModel for 1 epoch with shuffled DataLoader (Adam, MSE loss).
        5. Evaluate without shuffle to produce ordered per-window RMSE.

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
        self.model = Conv2DModel(n_visible, self.n_hidden, seq_len).to(self.device)
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
        """Train the Conv2D model on a batch of packet features.

        Computes per-feature min/max, normalizes, creates sliding windows,
        trains for 1 epoch with shuffled DataLoader, then evaluates without
        shuffle. Saves back_window for execution continuity.

        Args:
            data: Training data of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Per-window RMSE of shape ``(N - seq_len + 1,)``.
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
        """Score new data using the trained Conv2D model.

        Normalizes input, prepends the saved back_window for sliding-window
        continuity, computes per-window RMSE without shuffle, and trims
        output to match input length. Updates back_window for subsequent calls.

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
