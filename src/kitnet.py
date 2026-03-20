"""
KitNET: An Ensemble of Autoencoders for Online Network Intrusion Detection.

Three-phase pipeline:
  Phase 1 (Feature Mapping): Feed FM_GRACE_PERIOD samples to clustering
      -> produces a feature map assigning features to ensemble autoencoders.
  Phase 2 (AD Training): Feed AD_GRACE_PERIOD samples to the ensemble
      -> each autoencoder learns to reconstruct its feature subset.
      -> output layer learns to aggregate ensemble RMSE scores.
  Phase 3 (Execution): Score remaining samples.

Supports autoencoder types: 'elm', 'conv1d', 'conv2d', 'transformer',
'deep_mlp', 'lstm', 'stat'.

Clustering: 'corr' (hierarchical), 'dbscan', 'kmeans'.
Output layer: 'elm' (default) or 'stat' (statistical).

BUG FIX #3: Output layer receives properly shaped (N, n_aes) RMSE matrix.
"""
import logging
import numpy as np
from .clustering import get_clustering
from .autoencoders.elm import ELMAutoencoder
from .autoencoders.conv1d_ae import Conv1DAutoencoder
from .autoencoders.transformer_ae import TransformerAutoencoder
from .autoencoders.conv2d_ae import Conv2DAutoencoder
from .autoencoders.deep_mlp_ae import DeepMLPAutoencoder
try:
    from tf.lstm_ae import LSTMAutoencoder
except ImportError:
    LSTMAutoencoder = None
from .autoencoders.statistical_ae import StatisticalAnomaly
from . import config

log = logging.getLogger(__name__)

# AE types that use windowed/batch training (not packet-by-packet)
_DL_TYPES = {'conv1d', 'conv2d', 'transformer', 'deep_mlp', 'lstm'}


class KitNET:
    """
    KitNET ensemble of autoencoders.

    Args:
        n_features: Number of input features (e.g. 115).
        ae_type: 'elm', 'conv1d', 'conv2d', 'transformer', 'deep_mlp', 'lstm', 'stat'.
        clustering: 'corr', 'dbscan', or 'kmeans'.
        output_ae_type: 'elm' or 'stat'. Type of output aggregation layer.
        ar: If True, use autoregressive mode (DL models only).
        max_ae_size: Maximum features per ensemble autoencoder.
        hidden_ratio: Compression ratio for hidden layers.
        lr: Learning rate.
        fm_grace: Number of samples for feature mapping phase.
        ad_grace: Number of samples for AD training phase.
        exec_window: Batch size for execution phase.
        seq_len: Window size for windowed variants.
        device: PyTorch device for DL variants.
    """

    def __init__(self, n_features: int, ae_type: str = 'elm',
                 clustering: str = config.DEFAULT_CLUSTERING,
                 output_ae_type: str = config.DEFAULT_OUTPUT_AE,
                 ar: bool = False,
                 max_ae_size: int = config.MAX_AE_SIZE,
                 hidden_ratio: float = config.HIDDEN_RATIO,
                 lr: float = None,
                 fm_grace: int = config.FM_GRACE_PERIOD,
                 ad_grace: int = config.AD_GRACE_PERIOD,
                 exec_window: int = config.EXECUTION_WINDOW,
                 seq_len: int = config.SEQUENCE_LENGTH,
                 device: str = config.DEVICE):
        self.n_features = n_features
        self.ae_type = ae_type
        self.clustering_method = clustering
        self.output_ae_type = output_ae_type
        self.ar = ar
        self.max_ae_size = max_ae_size
        self.hidden_ratio = hidden_ratio
        self.fm_grace = fm_grace
        self.ad_grace = ad_grace
        self.exec_window = exec_window
        self.seq_len = seq_len
        self.device = device

        if lr is None:
            self.lr = config.LEARNING_RATE
        else:
            self.lr = lr

        self.clusterer = get_clustering(clustering, n_features)
        self.feature_map = None
        self.ensemble = []
        self.output_ae = None

    def _make_ae(self, n_visible: int):
        """Factory: create an autoencoder of the configured type."""
        if self.ae_type == 'elm':
            return ELMAutoencoder(n_visible, self.hidden_ratio, self.lr)
        elif self.ae_type == 'stat':
            return StatisticalAnomaly(n_visible)
        elif self.ae_type == 'conv1d':
            return Conv1DAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                     self.seq_len, self.device)
        elif self.ae_type == 'conv2d':
            return Conv2DAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                     self.seq_len, self.device)
        elif self.ae_type == 'transformer':
            return TransformerAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                          self.seq_len, self.device)
        elif self.ae_type == 'deep_mlp':
            return DeepMLPAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                      self.seq_len, self.device)
        elif self.ae_type == 'lstm':
            return LSTMAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                    self.seq_len, ar=self.ar, device=self.device)
        else:
            raise ValueError(f"Unknown ae_type: {self.ae_type}")

    def _make_output_ae(self, n_inputs: int):
        """Create the output aggregation layer."""
        if self.output_ae_type == 'stat':
            return StatisticalAnomaly(n_inputs)
        else:
            return ELMAutoencoder(
                n_visible=n_inputs,
                hidden_ratio=self.hidden_ratio,
                lr=config.LEARNING_RATE,
            )

    def _build_ensemble(self):
        """Build ensemble autoencoders from the feature map."""
        self.ensemble = [self._make_ae(len(fmap)) for fmap in self.feature_map]
        self.output_ae = self._make_output_ae(len(self.ensemble))
        n_hidden = getattr(self.output_ae, 'n_hidden', '?')
        log.info(f"Built ensemble: {len(self.ensemble)} AEs, "
                 f"output layer ({self.output_ae_type}): {len(self.ensemble)} -> {n_hidden}")

    def run(self, X: np.ndarray):
        """
        Run the full KitNET pipeline on dataset X of shape (N, n_features).

        Returns:
            scores: np.ndarray of anomaly scores for the execution phase.
        """
        N = len(X)
        total_train = self.fm_grace + self.ad_grace

        # === Phase 1: Feature Mapping ===
        log.info(f"Phase 1: Feature mapping ({self.fm_grace} samples, method={self.clustering_method})...")
        for i in range(min(self.fm_grace, N)):
            self.clusterer.update(X[i])
        self.feature_map = self.clusterer.cluster(self.max_ae_size)
        self._build_ensemble()
        log.info(f"Feature map: {self.n_features} features -> "
                 f"{len(self.feature_map)} clusters "
                 f"(sizes: {[len(f) for f in self.feature_map]})")

        # === Phase 2: AD Training ===
        train_start = self.fm_grace
        train_end = min(train_start + self.ad_grace, N)
        train_data = X[train_start:train_end]
        log.info(f"Phase 2: AD training ({len(train_data)} samples, type={self.ae_type})...")

        if self.ae_type in _DL_TYPES:
            self._train_dl(train_data)
        else:
            self._train_elm(train_data)

        # === Phase 3: Execution ===
        exec_start = total_train
        if exec_start >= N:
            log.warning("No data left for execution phase!")
            return np.array([])

        exec_data = X[exec_start:]
        log.info(f"Phase 3: Execution ({len(exec_data)} samples)...")
        scores = self._execute(exec_data)
        log.info(f"Execution complete. {len(scores)} scores produced.")
        return scores

    def _train_elm(self, train_data: np.ndarray) -> np.ndarray:
        """
        ELM/stat training: process sample-by-sample through each ensemble AE.
        Then train the output layer on the RMSE matrix.
        """
        n_aes = len(self.ensemble)
        N = len(train_data)
        rmse_matrix = np.zeros((N, n_aes))

        for i, fmap in enumerate(self.feature_map):
            subset = train_data[:, fmap]
            rmse_matrix[:, i] = self.ensemble[i].train(subset)
            if (i + 1) % 10 == 0:
                log.info(f"  Ensemble AE {i+1}/{n_aes} trained")

        log.info(f"  Training output layer on RMSE matrix shape {rmse_matrix.shape}")
        self.output_ae.train(rmse_matrix)
        return rmse_matrix

    def _train_dl(self, train_data: np.ndarray) -> np.ndarray:
        """
        DL training: batch train each ensemble AE.
        Then train output layer on the ensemble RMSE matrix.
        """
        n_aes = len(self.ensemble)
        rmse_list = []

        for i, fmap in enumerate(self.feature_map):
            subset = train_data[:, fmap]
            rmse = self.ensemble[i].train(subset)
            rmse_list.append(rmse)
            log.info(f"  DL ensemble AE {i+1}/{n_aes} trained, "
                     f"RMSE shape: {rmse.shape}")

        lengths = [len(r) for r in rmse_list]
        min_len = min(lengths)
        if max(lengths) != min_len:
            log.warning(f"  RMSE length mismatch: {lengths}. Truncating to {min_len}.")
        rmse_matrix = np.column_stack([r[:min_len] for r in rmse_list])

        log.info(f"  Training output layer on RMSE matrix shape {rmse_matrix.shape}")
        self.output_ae.train(rmse_matrix)
        return rmse_matrix

    def _execute(self, data: np.ndarray) -> np.ndarray:
        """Execute in batches of exec_window size."""
        all_scores = []
        n_aes = len(self.ensemble)
        is_dl = self.ae_type in _DL_TYPES

        for start in range(0, len(data), self.exec_window):
            end = min(start + self.exec_window, len(data))
            batch = data[start:end]

            if is_dl:
                rmse_list = []
                for i, fmap in enumerate(self.feature_map):
                    rmse = self.ensemble[i].execute(batch[:, fmap])
                    rmse_list.append(rmse)
                min_len = min(len(r) for r in rmse_list)
                rmse_matrix = np.column_stack([r[:min_len] for r in rmse_list])
            else:
                rmse_matrix = np.zeros((len(batch), n_aes))
                for i, fmap in enumerate(self.feature_map):
                    rmse_matrix[:, i] = self.ensemble[i].execute(batch[:, fmap])

            scores = self.output_ae.execute(rmse_matrix)
            all_scores.append(scores)

            if (start // self.exec_window) % 50 == 0:
                log.info(f"  Executed {end}/{len(data)} samples")

        return np.concatenate(all_scores)
