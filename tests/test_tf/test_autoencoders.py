"""Tests for all TensorFlow autoencoder variants: ELM, Statistical, Conv1D, Conv2D, Transformer, DeepMLP, LSTM."""
import numpy as np
import pytest
import tensorflow as tf

from src.common.autoencoders.elm import ELMAutoencoder
from src.common.autoencoders.statistical_ae import StatisticalAnomaly
from src.tf.autoencoders.conv1d_ae import Conv1DAutoencoder
from src.tf.autoencoders.conv2d_ae import Conv2DAutoencoder
from src.tf.autoencoders.transformer_ae import TransformerAutoencoder
from src.tf.autoencoders.deep_mlp_ae import DeepMLPAutoencoder
from src.tf.autoencoders.lstm_ae import LSTMAutoencoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n_samples, n_features, seed=42):
    """Synthetic data with structure (not random noise)."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, n_samples)
    base = np.column_stack([np.sin(t + i) for i in range(n_features)])
    noise = rng.randn(n_samples, n_features) * 0.1
    return (base + noise).astype(np.float32)


def _make_anomaly_data(n_samples, n_features, seed=99):
    """Data that is very different from _make_data — should score higher."""
    rng = np.random.RandomState(seed)
    return (rng.randn(n_samples, n_features) * 10 + 50).astype(np.float32)


# ---------------------------------------------------------------------------
# ELM Autoencoder (shared, pure NumPy)
# ---------------------------------------------------------------------------

class TestELMAutoencoder:

    def test_train_returns_nonzero_rmse(self):
        data = _make_data(200, 5)
        ae = ELMAutoencoder(5)
        rmse = ae.train(data)
        assert rmse.shape == (200,)
        assert np.all(rmse >= 0)
        assert np.sum(rmse > 0) > 100, "Most RMSE values should be > 0"

    def test_execute_returns_nonzero(self):
        data = _make_data(200, 5)
        ae = ELMAutoencoder(5)
        ae.train(data)
        rmse = ae.execute(data[:50])
        assert rmse.shape == (50,)
        assert np.all(rmse >= 0)
        assert np.mean(rmse) > 0

    def test_anomaly_scores_higher(self):
        train = _make_data(500, 5)
        ae = ELMAutoencoder(5, lr=0.1)
        ae.train(train)
        normal_rmse = ae.execute(train[:100])
        anomaly_rmse = ae.execute(_make_anomaly_data(100, 5))
        assert np.mean(anomaly_rmse) > np.mean(normal_rmse)


# ---------------------------------------------------------------------------
# Statistical Anomaly (shared, pure NumPy)
# ---------------------------------------------------------------------------

class TestStatisticalAnomaly:

    def test_train_returns_scores(self):
        data = _make_data(100, 5)
        ae = StatisticalAnomaly(5)
        scores = ae.train(data)
        assert scores.shape == (100,)
        assert np.all(scores >= 0)
        assert np.mean(scores) > 0

    def test_anomaly_scores_higher(self):
        train = _make_data(200, 5)
        ae = StatisticalAnomaly(5)
        ae.train(train)
        normal = ae.execute(train[:50])
        anomaly = ae.execute(_make_anomaly_data(50, 5))
        assert np.mean(anomaly) > np.mean(normal)


# ---------------------------------------------------------------------------
# DL Autoencoders (Conv1D, Conv2D, Transformer, DeepMLP, LSTM)
# ---------------------------------------------------------------------------

DL_SEQ_LEN = 10
DL_N_FEATURES = 4
DL_TRAIN_SAMPLES = 50
DL_DEVICE = 'cpu'


def _dl_train_data():
    return _make_data(DL_TRAIN_SAMPLES, DL_N_FEATURES)


def _dl_anomaly_data():
    return _make_anomaly_data(20, DL_N_FEATURES)


class _DLAutoencoderTestBase:
    """Shared tests for all TF DL autoencoders. Subclasses set ae_class."""
    ae_class = None

    def _make_ae(self, n_visible=DL_N_FEATURES, seq_len=DL_SEQ_LEN):
        return self.ae_class(n_visible, hidden_ratio=0.75, lr=0.001,
                             seq_len=seq_len, device=DL_DEVICE)

    def test_train_shape(self):
        ae = self._make_ae()
        data = _dl_train_data()
        rmse = ae.train(data)
        expected_len = DL_TRAIN_SAMPLES - DL_SEQ_LEN + 1
        assert rmse.shape == (expected_len,), f"Expected ({expected_len},), got {rmse.shape}"

    def test_train_nonzero(self):
        ae = self._make_ae()
        rmse = ae.train(_dl_train_data())
        assert np.all(rmse >= 0)
        assert np.mean(rmse) > 0, "Mean RMSE should not be zero"
        assert np.sum(rmse == 0) < len(rmse) * 0.5, "Too many exact zeros"

    def test_execute_shape(self):
        ae = self._make_ae()
        ae.train(_dl_train_data())
        exec_rmse = ae.execute(_dl_train_data())
        assert exec_rmse.shape == (DL_TRAIN_SAMPLES,), \
            f"Execute should return N scores, got {exec_rmse.shape}"

    def test_execute_nonzero(self):
        ae = self._make_ae()
        ae.train(_dl_train_data())
        rmse = ae.execute(_dl_train_data())
        assert np.mean(rmse) > 0

    def test_execute_small_batch(self):
        """Execute with fewer samples than seq_len (uses back_window)."""
        ae = self._make_ae()
        ae.train(_dl_train_data())
        small = _make_data(5, DL_N_FEATURES)
        rmse = ae.execute(small)
        assert len(rmse) == 5
        assert np.all(np.isfinite(rmse))

    def test_execute_very_small_returns_zeros(self):
        """If back_window is None and data < seq_len, return zeros."""
        ae = self._make_ae()
        ae.train(_dl_train_data())
        ae.back_window = None
        small = _make_data(5, DL_N_FEATURES)
        rmse = ae.execute(small)
        assert len(rmse) == 5
        np.testing.assert_array_equal(rmse, 0.0)

    def test_normalization_stored(self):
        ae = self._make_ae()
        data = _dl_train_data()
        ae.train(data)
        np.testing.assert_array_equal(ae.norm_min, np.min(data, axis=0))
        np.testing.assert_array_equal(ae.norm_max, np.max(data, axis=0))

    def test_back_window_saved(self):
        ae = self._make_ae()
        ae.train(_dl_train_data())
        assert ae.back_window is not None
        assert ae.back_window.shape == (DL_SEQ_LEN - 1, DL_N_FEATURES)

    def test_consecutive_execute(self):
        """Two execute calls should both produce valid results."""
        ae = self._make_ae()
        ae.train(_dl_train_data())
        r1 = ae.execute(_make_data(20, DL_N_FEATURES, seed=1))
        r2 = ae.execute(_make_data(20, DL_N_FEATURES, seed=2))
        assert len(r1) == 20
        assert len(r2) == 20
        assert np.all(np.isfinite(r1))
        assert np.all(np.isfinite(r2))

    def test_single_feature(self):
        """Must work with n_visible=1."""
        ae = self._make_ae(n_visible=1)
        data = _make_data(30, 1)
        rmse = ae.train(data)
        assert len(rmse) == 30 - DL_SEQ_LEN + 1
        assert np.mean(rmse) > 0

    def test_exact_seq_len_input(self):
        """Input of exactly seq_len samples -> 1 window."""
        ae = self._make_ae()
        data = _make_data(DL_SEQ_LEN, DL_N_FEATURES)
        rmse = ae.train(data)
        assert rmse.shape == (1,)
        assert rmse[0] > 0

    def test_no_nan_on_constant_input(self):
        """Constant input should not produce NaN."""
        data = np.full((DL_TRAIN_SAMPLES, DL_N_FEATURES), 3.14, dtype=np.float32)
        ae = self._make_ae()
        rmse = ae.train(data)
        assert np.all(np.isfinite(rmse))


class TestConv1DAutoencoder(_DLAutoencoderTestBase):
    ae_class = Conv1DAutoencoder


class TestConv2DAutoencoder(_DLAutoencoderTestBase):
    ae_class = Conv2DAutoencoder


class TestTransformerAutoencoder(_DLAutoencoderTestBase):
    ae_class = TransformerAutoencoder


class TestDeepMLPAutoencoder(_DLAutoencoderTestBase):
    ae_class = DeepMLPAutoencoder


class TestLSTMAutoencoder(_DLAutoencoderTestBase):
    ae_class = LSTMAutoencoder


# ---------------------------------------------------------------------------
# Cross-variant: anomaly sensitivity
# ---------------------------------------------------------------------------

class TestAnomalySensitivity:
    """All TF DL variants must score anomalous data higher than training data."""

    @pytest.mark.parametrize("ae_cls", [
        Conv1DAutoencoder, Conv2DAutoencoder, TransformerAutoencoder,
        DeepMLPAutoencoder, LSTMAutoencoder
    ])
    def test_anomaly_higher_than_normal(self, ae_cls):
        train = _make_data(60, DL_N_FEATURES, seed=0)
        ae = ae_cls(DL_N_FEATURES, hidden_ratio=0.75, lr=0.01,
                    seq_len=DL_SEQ_LEN, device=DL_DEVICE)
        ae.train(train)

        normal_rmse = ae.execute(train)
        anomaly_rmse = ae.execute(_make_anomaly_data(60, DL_N_FEATURES))

        assert np.mean(anomaly_rmse) > np.mean(normal_rmse) * 0.5, \
            f"{ae_cls.__name__}: anomaly mean {np.mean(anomaly_rmse):.4f} " \
            f"not clearly higher than normal {np.mean(normal_rmse):.4f}"
