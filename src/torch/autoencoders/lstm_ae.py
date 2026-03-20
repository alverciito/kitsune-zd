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
    """LSTM encoder-decoder for sequence-to-one anomaly detection.

    Architecture (Section III-C of the paper):
        Encoder: LSTM(n_visible, n_hidden, batch_first=True)
            Processes the full input sequence and extracts the final
            hidden state as a fixed-length representation.
        Decoder: Linear(n_hidden, n_hidden) + ReLU
                 -> Linear(n_hidden, n_visible) + Sigmoid

    Unlike the Conv1D/Conv2D/Transformer models that reconstruct entire
    windows, this model predicts a single frame from the sequence:
        - TSR mode (ar=False): predicts the last frame of the window.
        - AR mode (ar=True): predicts the next frame after the window.

    Input shape:
        ``(batch, seq_len, n_visible)`` - batch of sliding windows.
    Output shape:
        ``(batch, n_visible)`` - predicted single frame.

    Args:
        n_visible: Number of input features per timestep.
        n_hidden: LSTM hidden state size and decoder intermediate size.
    """

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
        """Forward pass: encode sequence via LSTM, decode final hidden state.

        The LSTM processes the full input sequence. Only the final hidden
        state ``h_n`` is used (sequence-to-one), which is then decoded
        to predict a single output frame.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_visible)``.

        Returns:
            torch.Tensor: Predicted frame of shape ``(batch, n_visible)``.
        """
        # x: (batch, seq_len, n_visible)
        _, (h_n, _) = self.encoder(x)  # h_n: (1, batch, n_hidden)
        h = h_n.squeeze(0)  # (batch, n_hidden)
        return self.decoder(h)  # (batch, n_visible)


class LSTMAutoencoder:
    """Windowed LSTM autoencoder with online normalization.

    Wraps _LSTMModel with sliding-window preprocessing, incremental
    min-max normalization, and back_window continuity for streaming
    execution across chunks.

    Supports two windowing strategies:
        - TSR mode (ar=False): Temporal Sequence Reconstruction. Each
          window ``x[i:i+seq_len]`` is used to predict its last frame
          ``x[i+seq_len-1]``. Measures reconstruction accuracy.
        - AR mode (ar=True): Autoregressive. Each window
          ``x[i:i+seq_len]`` is used to predict the next frame
          ``x[i+seq_len]``. Measures predictive accuracy.

    Training procedure:
        1. Compute per-feature min/max from training data.
        2. Normalize to [0, 1] via min-max scaling.
        3. Create (window, target) pairs using ``_make_windows_and_targets``.
        4. Train _LSTMModel for 1 epoch with shuffled DataLoader (Adam, MSE loss).
        5. Evaluate without shuffle to produce ordered per-window RMSE.

    Args:
        n_visible: Number of input features per packet.
        hidden_ratio: Compression ratio for LSTM hidden size (default: 0.75).
        lr: Learning rate for Adam optimizer (default: 0.001).
        seq_len: Sliding window length in packets (default: 500).
        ar: If True, use autoregressive windowing (predict next frame).
            If False, use TSR windowing (predict last frame).
        device: PyTorch device string ('cuda' or 'cpu').
    """

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
        """Apply min-max normalization to scale features to [0, 1].

        Args:
            x: Input array of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Normalized array of same shape, values in [0, 1].
        """
        return (x - self.norm_min) / (self.norm_max - self.norm_min + EPSILON)

    def _make_windows_and_targets(self, x_norm: np.ndarray):
        """Create (window, target) pairs based on the windowing strategy.

        Generates sliding windows and their corresponding single-frame
        targets. The target selection depends on the mode:
            - TSR mode (ar=False): target is the last frame of the window,
              ``x[i+seq_len-1]``.
            - AR mode (ar=True): target is the frame immediately after
              the window, ``x[i+seq_len]``, requiring one additional
              sample beyond the window end.

        Args:
            x_norm: Normalized input array of shape ``(N, n_visible)``.

        Returns:
            tuple: A pair ``(windows, targets)`` where:
                - windows: np.ndarray of shape ``(n_windows, seq_len, n_visible)``
                - targets: np.ndarray of shape ``(n_windows, n_visible)``
                Returns empty arrays if insufficient data for any windows.
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
        """Train the LSTM model on a batch of packet features.

        Computes per-feature min/max, normalizes, creates (window, target)
        pairs via ``_make_windows_and_targets``, trains for 1 epoch with a
        shuffled DataLoader (Adam optimizer, MSE loss), then evaluates
        without shuffle. The per-window RMSE is computed as the
        root-mean-square error between predicted and target frames,
        averaged over the n_visible features.

        Saves the last ``seq_len - 1`` normalized samples as back_window
        for execution continuity.

        Args:
            data: Training data of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Per-window RMSE array. Length depends on mode:
                TSR: ``N - seq_len + 1``, AR: ``N - seq_len``.
                Returns empty array if insufficient data.
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
        """Score new data using the trained LSTM model.

        Normalizes input, prepends the saved back_window for sliding-window
        continuity, creates (window, target) pairs, computes per-window
        RMSE without shuffle, and trims output to match input length.
        Updates back_window for subsequent calls.

        Args:
            data: Input data of shape ``(N, n_visible)``.

        Returns:
            np.ndarray: Per-sample RMSE scores of shape ``(N,)``.
                Returns zeros if extended data is shorter than seq_len
                or if no valid windows can be created.
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
