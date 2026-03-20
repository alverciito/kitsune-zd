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
    """KitNET: An Ensemble of Autoencoders for Online Network Intrusion Detection.

    Implements the three-phase KitNET pipeline (Section III of the paper):
        Phase 1 (Feature Mapping): Clusters correlated features into subsets
            using hierarchical correlation clustering, DBSCAN, or K-Means.
        Phase 2 (AD Training): Trains one autoencoder per feature cluster
            to learn normal traffic reconstruction, then trains an output
            aggregation layer on the ensemble RMSE scores.
        Phase 3 (Execution): Scores new samples by computing per-AE RMSE
            and aggregating through the output layer.

    The ensemble supports multiple autoencoder backends: lightweight ELM
    for packet-by-packet processing, or deep-learning variants (Conv1D,
    Conv2D, Transformer, DeepMLP, LSTM) that operate on sliding windows
    of packets. A statistical baseline ('stat') is also available.

    Attributes:
        feature_map: List of index arrays mapping features to ensemble AEs,
            populated after Phase 1.
        ensemble: List of trained autoencoder instances, one per cluster.
        output_ae: Aggregation layer (ELM or statistical) that converts
            per-AE RMSE vectors into a single anomaly score.

    Args:
        n_features: Number of input features per sample (e.g., 115 for
            the AfterImage feature extractor).
        ae_type: Autoencoder backend type. One of 'elm', 'conv1d',
            'conv2d', 'transformer', 'deep_mlp', 'lstm', 'stat'.
        clustering: Feature clustering algorithm. One of 'corr'
            (hierarchical correlation), 'dbscan', or 'kmeans'.
        output_ae_type: Output aggregation layer type. 'elm' (default)
            for a single-layer ELM, or 'stat' for statistical scoring.
        ar: If True, use autoregressive windowing for DL models
            (predict next frame instead of reconstructing last frame).
            Only applicable to DL ae_types.
        max_ae_size: Maximum number of features assigned to a single
            ensemble autoencoder during clustering.
        hidden_ratio: Compression ratio for hidden layer size
            (default: 0.22, Table II of the paper).
        lr: Learning rate for Adam optimizer (DL types) or ELM.
            Defaults to config.LEARNING_RATE if None.
        fm_grace: Number of samples consumed during Phase 1
            (Feature Mapping grace period).
        ad_grace: Number of samples consumed during Phase 2
            (Anomaly Detection training grace period).
        exec_window: Batch size for Phase 3 execution. Controls memory
            usage during scoring.
        seq_len: Sliding window length in packets for DL variants
            (default: 800, Table II of the paper).
        device: PyTorch device string ('cuda' or 'cpu') for DL variants.
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
        """Create an autoencoder of the configured type for an ensemble slot.

        Factory method that instantiates the appropriate autoencoder class
        based on ``self.ae_type``, forwarding relevant hyperparameters
        (hidden_ratio, lr, seq_len, device, ar).

        Args:
            n_visible: Number of input features for this ensemble member,
                determined by the clustering step.

        Returns:
            An autoencoder instance with ``train()`` and ``execute()`` methods.

        Raises:
            ValueError: If ``self.ae_type`` is not a recognized type.
        """
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
        """Create the output aggregation layer for ensemble score fusion.

        The output layer receives a vector of per-AE RMSE scores (one per
        ensemble member) and produces a single anomaly score. Supports ELM
        (default) or statistical aggregation.

        Args:
            n_inputs: Number of ensemble autoencoders, i.e., the
                dimensionality of the RMSE vector fed to this layer.

        Returns:
            An autoencoder instance (ELMAutoencoder or StatisticalAnomaly)
            with ``train()`` and ``execute()`` methods.
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
        """Build ensemble autoencoders and output layer from the feature map.

        Iterates over ``self.feature_map`` (produced by Phase 1 clustering)
        and creates one autoencoder per feature cluster via ``_make_ae``.
        Also instantiates the output aggregation layer sized to accept
        one RMSE score per ensemble member.

        Side effects:
            Populates ``self.ensemble`` and ``self.output_ae``.
        """
        self.ensemble = [self._make_ae(len(fmap)) for fmap in self.feature_map]
        self.output_ae = self._make_output_ae(len(self.ensemble))
        n_hidden = getattr(self.output_ae, 'n_hidden', '?')
        log.info(f"Built ensemble: {len(self.ensemble)} AEs, "
                 f"output layer ({self.output_ae_type}): {len(self.ensemble)} -> {n_hidden}")

    def run(self, X: np.ndarray):
        """Run the full three-phase KitNET pipeline on a dataset.

        Sequentially executes:
            1. Phase 1 (Feature Mapping): Feeds the first ``fm_grace``
               samples to the clusterer, then generates the feature map.
            2. Phase 2 (AD Training): Trains all ensemble AEs and the
               output layer on the next ``ad_grace`` samples.
            3. Phase 3 (Execution): Scores all remaining samples.

        Args:
            X: Input data array of shape ``(N, n_features)`` where N is
                the total number of samples. Must have at least
                ``fm_grace + ad_grace`` samples for training; any
                remainder is scored in the execution phase.

        Returns:
            np.ndarray: Anomaly scores for the execution-phase samples,
                shape ``(N - fm_grace - ad_grace,)``. Returns an empty
                array if no data remains after the training phases.
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
        """Phase 2 training path for ELM and statistical autoencoders.

        Trains each ensemble AE on its feature subset by passing the
        full training data in one call (ELM/stat AEs handle batching
        internally). Collects per-AE RMSE scores into a matrix and
        trains the output aggregation layer on it.

        Args:
            train_data: Training samples of shape ``(N, n_features)``.

        Returns:
            np.ndarray: RMSE matrix of shape ``(N, n_aes)`` produced
                during ensemble training, used to train the output layer.
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
        """Phase 2 training path for deep-learning autoencoders.

        Batch-trains each ensemble AE (Conv1D, Conv2D, Transformer,
        DeepMLP, LSTM) on its feature subset. Because windowed AEs may
        produce different RMSE lengths (due to window truncation), the
        resulting vectors are truncated to the minimum length before
        column-stacking into the RMSE matrix. The output aggregation
        layer is then trained on this matrix.

        Args:
            train_data: Training samples of shape ``(N, n_features)``.

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
        """Phase 3: score data in batches through the trained ensemble.

        Processes the execution-phase data in chunks of ``exec_window``
        size to control memory usage. For each chunk:
            1. Each ensemble AE scores its feature subset, producing
               per-sample RMSE values.
            2. RMSE vectors are stacked into a matrix of shape
               ``(chunk_len, n_aes)``.
            3. The output layer scores the RMSE matrix to produce
               final anomaly scores.

        For DL types, the back_window mechanism in each AE ensures
        sliding-window continuity across chunk boundaries.

        Args:
            data: Execution-phase samples of shape ``(N, n_features)``.

        Returns:
            np.ndarray: Anomaly scores of shape ``(N,)`` (may be slightly
                shorter for DL types due to windowing edge effects).
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
