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
from ...common.utils import create_windows
from ...common.config import EPSILON, BATCH_SIZE


class Conv1DModel(nn.Module):
    """Conv1D encoder-decoder for windowed anomaly detection.

    Architecture (Section III-C of the paper):
        Encoder: Conv1D(n_visible, 3, kernel_size=seq_len, padding='same') + ReLU
                 -> Linear(3, n_hidden) + ReLU
        Decoder: Linear(n_hidden, n_visible) + ReLU

    The Conv1D layer operates along the temporal axis, compressing
    ``n_visible`` input channels to 3 intermediate channels at each
    timestep. The dense layers then reduce to the bottleneck and
    reconstruct the original feature dimensionality.

    Input shape:
        ``(batch, seq_len, n_visible)`` - a batch of sliding windows.
    Output shape:
        ``(batch, seq_len, n_visible)`` - reconstructed windows.

    Args:
        n_visible: Number of input features per timestep.
        n_hidden: Bottleneck dimensionality (typically ``ceil(n_visible * hidden_ratio)``).
        seq_len: Length of the input sliding window in timesteps.
    """

    def __init__(self, n_visible: int, n_hidden: int, seq_len: int):
        super().__init__()
        # Conv1D: PyTorch expects (batch, channels, length)
        # Original TF: Conv1D(filters=3, kernel_size=seq_len, padding='same')
        self.conv = nn.Conv1d(n_visible, 3, kernel_size=seq_len, padding='same')
        self.dense1 = nn.Linear(3, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_visible)

    def forward(self, x):
        """Forward pass: encode then decode the input window.

        Permutes input to channels-first for Conv1D, applies the
        encoder (conv + dense), then decodes back to the original
        feature space.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_visible)``.

        Returns:
            torch.Tensor: Reconstructed tensor of shape
                ``(batch, seq_len, n_visible)``.
        """
        # x: (batch, seq_len, n_visible)
        x = x.permute(0, 2, 1)       # -> (batch, n_visible, seq_len)
        x = F.relu(self.conv(x))      # -> (batch, 3, seq_len)
        x = x.permute(0, 2, 1)       # -> (batch, seq_len, 3)
        x = F.relu(self.dense1(x))   # -> (batch, seq_len, n_hidden)
        x = F.relu(self.dense2(x))   # -> (batch, seq_len, n_visible)
        return x


class Conv1DAutoencoder:
    """Windowed Conv1D autoencoder with online normalization.

    Wraps Conv1DModel with sliding-window preprocessing, incremental
    min-max normalization, and back_window continuity for streaming
    execution across chunks. Uses temporal sequence reconstruction (TSR)
    windowing: each window ``x[i:i+seq_len]`` is reconstructed in full.

    Training procedure:
        1. Compute per-feature min/max from training data.
        2. Normalize to [0, 1] via min-max scaling.
        3. Create sliding windows of shape ``(N - seq_len + 1, seq_len, n_visible)``.
        4. Train Conv1DModel for 1 epoch with shuffled DataLoader (Adam, MSE loss).
        5. Evaluate without shuffle to produce ordered per-window RMSE.

    Execution procedure:
        1. Normalize using stored min/max.
        2. Prepend ``back_window`` (last ``seq_len - 1`` samples from previous
           call) for sliding-window continuity across chunk boundaries.
        3. Score windows without shuffle; return last ``N`` RMSE values.
        4. Save new ``back_window`` for the next call.

    Bug Fix #2: Evaluation DataLoader never shuffles, ensuring RMSE
    ordering matches the original sample order.

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
        self.model = Conv1DModel(n_visible, self.n_hidden, seq_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.criterion = nn.MSELoss()

        # Normalization (set during training)
        self.norm_min = None
        self.norm_max = None
        # Back window for continuity during execution
        self.back_window = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to scale features to [0, 1].

        Uses the per-feature min/max values computed during training.
        A small epsilon is added to the denominator to avoid division
        by zero for constant features.

        Args:
            x: Input array of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Normalized array of same shape, values in [0, 1].
        """
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def train(self, data: np.ndarray) -> np.ndarray:
        """Train the Conv1D model on a batch of packet features.

        Performs the full training pipeline:
            1. Compute and store per-feature min/max for normalization.
            2. Normalize data to [0, 1].
            3. Create sliding windows of shape
               ``(N - seq_len + 1, seq_len, n_visible)``.
            4. Train the model for 1 epoch using a shuffled DataLoader
               with Adam optimizer and MSE loss.
            5. Evaluate (without shuffle) to compute per-window RMSE.
            6. Save the last ``seq_len - 1`` normalized samples as
               ``back_window`` for execution continuity.

        Args:
            data: Training data of shape ``(N, n_visible)`` where N is
                the number of packets and n_visible is the feature count.

        Returns:
            np.ndarray: Per-window RMSE array of shape
                ``(N - seq_len + 1,)``, representing reconstruction
                error for each sliding window position.
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
        """Score new data using the trained Conv1D model.

        Normalizes the input using stored min/max, prepends the saved
        ``back_window`` from the previous train/execute call to maintain
        sliding-window continuity, then computes per-window RMSE without
        shuffling. The output is trimmed to exactly ``N`` scores
        (matching the input length) by discarding leading entries that
        correspond to the prepended back_window.

        After scoring, saves the last ``seq_len - 1`` normalized samples
        as the new ``back_window`` for subsequent calls.

        Bug Fix #2: The evaluation DataLoader never shuffles, ensuring
        RMSE values are aligned with their original sample positions.

        Args:
            data: Input data of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Per-sample RMSE scores of shape ``(N,)``.
                Returns zeros if the extended data (back_window + data)
                is shorter than ``seq_len``.
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
