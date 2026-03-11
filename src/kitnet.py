"""
KitNET: An Ensemble of Autoencoders for Online Network Intrusion Detection.

Three-phase pipeline:
  Phase 1 (Feature Mapping): Feed FM_GRACE_PERIOD samples to CorClust
      -> produces a feature map assigning features to ensemble autoencoders.
  Phase 2 (AD Training): Feed AD_GRACE_PERIOD samples to the ensemble
      -> each autoencoder learns to reconstruct its feature subset.
      -> output layer learns to aggregate ensemble RMSE scores.
  Phase 3 (Execution): Score remaining samples.

Supports three autoencoder types: 'elm', 'conv1d', 'transformer'.
The output layer is ALWAYS an ELM autoencoder (pure numpy).

BUG FIX #3: Output layer receives properly shaped (N, n_aes) RMSE matrix.
"""
import logging
import numpy as np
from .corclust import CorClust
from .autoencoders.elm import ELMAutoencoder
from .autoencoders.conv1d_ae import Conv1DAutoencoder
from .autoencoders.transformer_ae import TransformerAutoencoder
from .autoencoders.conv2d_ae import Conv2DAutoencoder
from .autoencoders.deep_mlp_ae import DeepMLPAutoencoder
from . import config

log = logging.getLogger(__name__)


class KitNET:
    """
    KitNET ensemble of autoencoders.

    Args:
        n_features: Number of input features (e.g. 115).
        ae_type: 'elm', 'conv1d', or 'transformer'.
        max_ae_size: Maximum features per ensemble autoencoder.
        hidden_ratio: Compression ratio for hidden layers.
        lr: Learning rate.
        fm_grace: Number of samples for feature mapping phase.
        ad_grace: Number of samples for AD training phase.
        exec_window: Batch size for execution phase.
        seq_len: Window size for Conv1D/Transformer.
        device: PyTorch device for DL variants.
    """

    def __init__(self, n_features: int, ae_type: str = 'elm',
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
        self.max_ae_size = max_ae_size
        self.hidden_ratio = hidden_ratio
        self.fm_grace = fm_grace
        self.ad_grace = ad_grace
        self.exec_window = exec_window
        self.seq_len = seq_len
        self.device = device

        if lr is None:
            self.lr = config.LEARNING_RATE_ELM if ae_type == 'elm' else config.LEARNING_RATE_DL
        else:
            self.lr = lr

        self.corclust = CorClust(n_features)
        self.feature_map = None
        self.ensemble = []
        self.output_ae = None

    def _make_ae(self, n_visible: int):
        """Factory: create an autoencoder of the configured type."""
        if self.ae_type == 'elm':
            return ELMAutoencoder(n_visible, self.hidden_ratio, self.lr)
        elif self.ae_type == 'conv1d':
            return Conv1DAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                     self.seq_len, self.device)
        elif self.ae_type == 'transformer':
            return TransformerAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                          self.seq_len, self.device)
        elif self.ae_type == 'conv2d':
            return Conv2DAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                     self.seq_len, self.device)
        elif self.ae_type == 'deep_mlp':
            return DeepMLPAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                      self.seq_len, self.device)
        else:
            raise ValueError(f"Unknown ae_type: {self.ae_type}")

    def _build_ensemble(self):
        """Build ensemble autoencoders from the feature map."""
        self.ensemble = [self._make_ae(len(fmap)) for fmap in self.feature_map]
        # Output layer is always ELM, input = number of ensemble autoencoders
        self.output_ae = ELMAutoencoder(
            n_visible=len(self.ensemble),
            hidden_ratio=self.hidden_ratio,
            lr=config.LEARNING_RATE_ELM,
        )
        log.info(f"Built ensemble: {len(self.ensemble)} AEs, "
                 f"output layer: {len(self.ensemble)} -> {self.output_ae.n_hidden}")

    def run(self, X: np.ndarray):
        """
        Run the full KitNET pipeline on dataset X of shape (N, n_features).

        Returns:
            scores: np.ndarray of anomaly scores for the execution phase.
                    Length depends on ae_type: N - fm_grace - ad_grace for ELM,
                    or slightly less for windowed variants.
        """
        N = len(X)
        total_train = self.fm_grace + self.ad_grace

        # === Phase 1: Feature Mapping ===
        log.info(f"Phase 1: Feature mapping ({self.fm_grace} samples)...")
        for i in range(min(self.fm_grace, N)):
            self.corclust.update(X[i])
        self.feature_map = self.corclust.cluster(self.max_ae_size)
        self._build_ensemble()
        log.info(f"Feature map: {self.n_features} features -> "
                 f"{len(self.feature_map)} clusters "
                 f"(sizes: {[len(f) for f in self.feature_map]})")

        # === Phase 2: AD Training ===
        train_start = self.fm_grace
        train_end = min(train_start + self.ad_grace, N)
        train_data = X[train_start:train_end]
        log.info(f"Phase 2: AD training ({len(train_data)} samples, type={self.ae_type})...")

        if self.ae_type == 'elm':
            scores_phase2 = self._train_elm(train_data)
        else:
            scores_phase2 = self._train_dl(train_data)

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
        ELM training: process sample-by-sample through each ensemble AE.
        Then train the output layer on the RMSE matrix.
        """
        n_aes = len(self.ensemble)
        N = len(train_data)
        rmse_matrix = np.zeros((N, n_aes))

        for i, fmap in enumerate(self.feature_map):
            subset = train_data[:, fmap]
            rmse_matrix[:, i] = self.ensemble[i].train(subset)
            if (i + 1) % 10 == 0:
                log.info(f"  ELM ensemble AE {i+1}/{n_aes} trained")

        # Train output layer on the RMSE matrix (Bug #3 fix: consistent shape)
        log.info(f"  Training output layer on RMSE matrix shape {rmse_matrix.shape}")
        self.output_ae.train(rmse_matrix)
        return rmse_matrix

    def _train_dl(self, train_data: np.ndarray) -> np.ndarray:
        """
        DL training (Conv1D/Transformer): batch train each ensemble AE.
        Then train output layer on the ensemble RMSE matrix.

        Each AE processes windowed data and returns RMSE of length
        (N - seq_len + 1), which is the same for all AEs.
        """
        n_aes = len(self.ensemble)
        rmse_list = []

        for i, fmap in enumerate(self.feature_map):
            subset = train_data[:, fmap]
            rmse = self.ensemble[i].train(subset)
            rmse_list.append(rmse)
            log.info(f"  DL ensemble AE {i+1}/{n_aes} trained, "
                     f"RMSE shape: {rmse.shape}")

        # Bug #3 fix: all RMSE arrays should have the same length.
        # Verify and stack into (M, n_aes) matrix.
        lengths = [len(r) for r in rmse_list]
        min_len = min(lengths)
        if max(lengths) != min_len:
            log.warning(f"  RMSE length mismatch: {lengths}. Truncating to {min_len}.")
        rmse_matrix = np.column_stack([r[:min_len] for r in rmse_list])

        # Train output layer
        log.info(f"  Training output layer on RMSE matrix shape {rmse_matrix.shape}")
        self.output_ae.train(rmse_matrix)
        return rmse_matrix

    def _execute(self, data: np.ndarray) -> np.ndarray:
        """Execute in batches of exec_window size."""
        all_scores = []
        n_aes = len(self.ensemble)

        for start in range(0, len(data), self.exec_window):
            end = min(start + self.exec_window, len(data))
            batch = data[start:end]

            if self.ae_type == 'elm':
                rmse_matrix = np.zeros((len(batch), n_aes))
                for i, fmap in enumerate(self.feature_map):
                    rmse_matrix[:, i] = self.ensemble[i].execute(batch[:, fmap])
            else:
                rmse_list = []
                for i, fmap in enumerate(self.feature_map):
                    rmse = self.ensemble[i].execute(batch[:, fmap])
                    rmse_list.append(rmse)
                # Align lengths
                min_len = min(len(r) for r in rmse_list)
                rmse_matrix = np.column_stack([r[:min_len] for r in rmse_list])

            scores = self.output_ae.execute(rmse_matrix)
            all_scores.append(scores)

            if (start // self.exec_window) % 50 == 0:
                log.info(f"  Executed {end}/{len(data)} samples")

        return np.concatenate(all_scores)
