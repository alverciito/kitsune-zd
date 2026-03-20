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
from ..common.clustering import get_clustering
from ..common.autoencoders.elm import ELMAutoencoder
from .autoencoders.conv1d_ae import Conv1DAutoencoder
from .autoencoders.transformer_ae import TransformerAutoencoder
from .autoencoders.conv2d_ae import Conv2DAutoencoder
from .autoencoders.deep_mlp_ae import DeepMLPAutoencoder
from .autoencoders.lstm_ae import LSTMAutoencoder
from ..common.autoencoders.statistical_ae import StatisticalAnomaly
from ..common import config

log = logging.getLogger(__name__)

# AE types that use windowed/batch training (not packet-by-packet)
_DL_TYPES = {'conv1d', 'conv2d', 'transformer', 'deep_mlp', 'lstm'}


class KitNET:
    """KitNET ensemble of autoencoders for online network intrusion detection.

    Implements the three-phase KitNET pipeline (Mirsky et al., 2018):
      1. Feature Mapping -- cluster correlated features into groups.
      2. AD Training -- train one autoencoder per feature group, then train
         an output aggregation layer on the ensemble RMSE scores.
      3. Execution -- score unseen traffic using reconstruction error.

    This is the TensorFlow backend. It shares the same public API as the
    PyTorch ``KitNET`` but delegates to TF/Keras autoencoder implementations
    for the deep-learning variants (conv1d, conv2d, transformer, deep_mlp,
    lstm). The ELM and statistical backends are framework-agnostic and
    shared across both backends.

    Attributes:
        feature_map: List of index arrays produced by clustering (set after
            Phase 1).
        ensemble: List of trained ensemble autoencoders (set after Phase 2).
        output_ae: Trained output aggregation autoencoder (set after Phase 2).

    Args:
        n_features: Number of input features (e.g. 115 for the Kitsune
            feature extractor).
        ae_type: Autoencoder type for the ensemble layer. One of ``'elm'``,
            ``'conv1d'``, ``'conv2d'``, ``'transformer'``, ``'deep_mlp'``,
            ``'lstm'``, or ``'stat'``.
        clustering: Feature-clustering algorithm. One of ``'corr'``
            (hierarchical correlation), ``'dbscan'``, or ``'kmeans'``.
        output_ae_type: Autoencoder type for the output aggregation layer.
            ``'elm'`` (default) or ``'stat'`` (statistical baseline).
        ar: If True, use autoregressive windowing for deep-learning
            variants (predict next frame instead of reconstructing last
            frame). Ignored for ``'elm'`` and ``'stat'``.
        max_ae_size: Maximum number of features any single ensemble
            autoencoder may receive.
        hidden_ratio: Compression ratio for hidden layers (default from
            config, typically 0.75 per Table II of the paper).
        lr: Learning rate for the Adam optimizer. If ``None``, uses the
            default from ``config.LEARNING_RATE``.
        fm_grace: Number of samples consumed during the feature-mapping
            phase (Phase 1).
        ad_grace: Number of samples consumed during autoencoder training
            (Phase 2).
        exec_window: Batch size used when scoring samples in Phase 3.
        seq_len: Sliding-window length in packets for windowed DL
            variants (default from config).
        device: Device string. Accepted for API compatibility with the
            PyTorch backend; ignored by TensorFlow.
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
        """Create an ensemble autoencoder of the configured type.

        Factory method that instantiates the appropriate autoencoder class
        based on ``self.ae_type``, forwarding common hyper-parameters
        (hidden_ratio, lr, seq_len, device).

        Args:
            n_visible: Number of input features for this autoencoder,
                determined by the size of the corresponding feature cluster.

        Returns:
            An autoencoder instance exposing ``train()`` and ``execute()``
            methods.

        Raises:
            ValueError: If ``self.ae_type`` is not a recognised type string.
        """
        if self.ae_type == 'elm':
            return ELMAutoencoder(n_visible, self.hidden_ratio, self.lr)
        elif self.ae_type == 'stat':
            return StatisticalAnomaly(n_visible)
        elif self.ae_type == 'conv1d':
            return Conv1DAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                     self.seq_len, device=self.device)
        elif self.ae_type == 'conv2d':
            return Conv2DAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                     self.seq_len, device=self.device)
        elif self.ae_type == 'transformer':
            return TransformerAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                          self.seq_len, device=self.device)
        elif self.ae_type == 'deep_mlp':
            return DeepMLPAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                      self.seq_len, device=self.device)
        elif self.ae_type == 'lstm':
            return LSTMAutoencoder(n_visible, self.hidden_ratio, self.lr,
                                    self.seq_len, ar=self.ar, device=self.device)
        else:
            raise ValueError(f"Unknown ae_type: {self.ae_type}")

    def _make_output_ae(self, n_inputs: int):
        """Create the output aggregation layer.

        The output layer receives the vector of per-autoencoder RMSE scores
        (one per ensemble member) and produces a single scalar anomaly
        score per sample.

        Args:
            n_inputs: Number of ensemble autoencoders, i.e. the
                dimensionality of the RMSE vector fed to the output layer.

        Returns:
            An autoencoder (ELM or StatisticalAnomaly) exposing ``train()``
            and ``execute()`` methods.
        """
        if self.output_ae_type == 'stat':
            return StatisticalAnomaly(n_inputs)
        else:
            return ELMAutoencoder(
                n_visible=n_inputs,
                hidden_ratio=self.hidden_ratio,
                lr=config.LEARNING_RATE,
            )

    def _build_ensemble(self):
        """Build ensemble autoencoders from the feature map.

        Iterates over ``self.feature_map`` (set during Phase 1) and creates
        one autoencoder per feature cluster via ``_make_ae``. Also creates
        the output aggregation layer whose input size equals the number of
        ensemble members.

        Must be called after ``self.feature_map`` has been populated by the
        clustering step.
        """
        self.ensemble = [self._make_ae(len(fmap)) for fmap in self.feature_map]
        self.output_ae = self._make_output_ae(len(self.ensemble))
        n_hidden = getattr(self.output_ae, 'n_hidden', '?')
        log.info(f"Built ensemble: {len(self.ensemble)} AEs, "
                 f"output layer ({self.output_ae_type}): {len(self.ensemble)} -> {n_hidden}")

    def run(self, X: np.ndarray):
        """Run the full three-phase KitNET pipeline.

        Sequentially executes:
          1. **Feature Mapping** -- consumes the first ``fm_grace`` rows to
             cluster features.
          2. **AD Training** -- consumes the next ``ad_grace`` rows to train
             the ensemble and output autoencoders.
          3. **Execution** -- scores all remaining rows.

        Args:
            X: Input dataset of shape ``(N, n_features)`` where *N* must
                be greater than ``fm_grace + ad_grace`` for any scores to
                be produced.

        Returns:
            np.ndarray: One-dimensional array of anomaly scores for the
            execution-phase samples (length ``N - fm_grace - ad_grace``).
            Returns an empty array if no data remains after training.
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
        """Train the ensemble using packet-by-packet (ELM/stat) autoencoders.

        Each ensemble autoencoder receives its feature subset of
        ``train_data`` and is trained via its ``train()`` method, which
        processes all samples at once and returns per-sample RMSE. The
        resulting RMSE matrix (shape ``(N, n_aes)``) is then used to
        train the output aggregation layer.

        Args:
            train_data: Training data of shape ``(N, n_features)``.

        Returns:
            np.ndarray: RMSE matrix of shape ``(N, n_aes)`` produced by
            the ensemble during training.
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
        """Train the ensemble using windowed deep-learning autoencoders.

        Each DL autoencoder internally creates sliding windows from its
        feature subset. Because windowing can produce different numbers
        of output RMSE values depending on feature-subset size, the
        per-AE RMSE arrays are truncated to the minimum common length
        before being column-stacked into the RMSE matrix (BUG FIX #3).

        The RMSE matrix is then used to train the output aggregation
        layer.

        Args:
            train_data: Training data of shape ``(N, n_features)``.

        Returns:
            np.ndarray: RMSE matrix of shape ``(min_len, n_aes)`` where
            ``min_len`` is the shortest per-AE RMSE vector length.
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
        """Score data through the trained ensemble and output layer.

        Processes ``data`` in non-overlapping batches of size
        ``self.exec_window``. For each batch, every ensemble autoencoder
        produces per-sample RMSE on its feature subset. The per-AE RMSE
        vectors are stacked into a matrix and passed through the output
        aggregation layer to yield final anomaly scores.

        For DL autoencoders, RMSE vectors may differ in length across
        ensemble members due to windowing; they are truncated to the
        minimum length before stacking.

        Args:
            data: Execution-phase data of shape ``(N, n_features)``.

        Returns:
            np.ndarray: One-dimensional array of anomaly scores,
            one per sample (or per window for DL types).
        """
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
