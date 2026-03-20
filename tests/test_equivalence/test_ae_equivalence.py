"""Cross-framework equivalence tests for autoencoders."""
import numpy as np
import pytest

# Framework-agnostic AEs (should be identical)
from src.common.autoencoders import ELMAutoencoder, StatisticalAnomaly


def _make_data(n, d, seed=42):
    rng = np.random.RandomState(seed)
    return rng.rand(n, d).astype(np.float32)


def _make_anomaly(n, d, seed=99):
    rng = np.random.RandomState(seed)
    return (rng.rand(n, d) * 5 + 3).astype(np.float32)


class TestELMIdentical:
    """ELM is pure NumPy -- both backends must produce identical results."""

    def test_elm_same_scores(self):
        data = _make_data(200, 4)
        ae1 = ELMAutoencoder(4, hidden_ratio=0.75)
        ae2 = ELMAutoencoder(4, hidden_ratio=0.75)
        r1 = ae1.train(data)
        r2 = ae2.train(data)
        np.testing.assert_array_equal(r1, r2)


class TestStatIdentical:
    """Statistical AE is pure NumPy -- must produce identical results."""

    def test_stat_same_scores(self):
        data = _make_data(200, 4)
        ae1 = StatisticalAnomaly(4)
        ae2 = StatisticalAnomaly(4)
        r1 = ae1.train(data)
        r2 = ae2.train(data)
        np.testing.assert_array_equal(r1, r2)


class TestDLEquivalence:
    """DL AEs from both backends should detect anomalies similarly."""

    @pytest.fixture(params=['conv1d', 'conv2d', 'transformer', 'deep_mlp', 'lstm'])
    def ae_type(self, request):
        return request.param

    def _get_ae(self, backend, ae_type, n_visible=4, seq_len=10):
        kwargs = dict(n_visible=n_visible, hidden_ratio=0.75,
                      lr=0.01, seq_len=seq_len, device='cpu')
        if backend == 'torch':
            from src.torch import autoencoders as torch_ae
            cls_map = {
                'conv1d': torch_ae.Conv1DAutoencoder,
                'conv2d': torch_ae.Conv2DAutoencoder,
                'transformer': torch_ae.TransformerAutoencoder,
                'deep_mlp': torch_ae.DeepMLPAutoencoder,
                'lstm': torch_ae.LSTMAutoencoder,
            }
        else:
            from src.tf import autoencoders as tf_ae
            cls_map = {
                'conv1d': tf_ae.Conv1DAutoencoder,
                'conv2d': tf_ae.Conv2DAutoencoder,
                'transformer': tf_ae.TransformerAutoencoder,
                'deep_mlp': tf_ae.DeepMLPAutoencoder,
                'lstm': tf_ae.LSTMAutoencoder,
            }
        return cls_map[ae_type](**kwargs)

    def test_output_shape_matches(self, ae_type):
        data = _make_data(60, 4)
        torch_ae = self._get_ae('torch', ae_type)
        tf_ae = self._get_ae('tf', ae_type)
        r_torch = torch_ae.train(data)
        r_tf = tf_ae.train(data)
        assert r_torch.shape == r_tf.shape

    def test_both_detect_anomaly(self, ae_type):
        train = _make_data(80, 4)
        anomaly = _make_anomaly(30, 4)
        for backend in ['torch', 'tf']:
            ae = self._get_ae(backend, ae_type)
            ae.train(train)
            normal_scores = ae.execute(train[-30:])
            anomaly_scores = ae.execute(anomaly)
            assert np.mean(anomaly_scores) > np.mean(normal_scores), \
                f"{backend}/{ae_type}: anomaly not detected"

    def test_rmse_same_order_of_magnitude(self, ae_type):
        data = _make_data(60, 4)
        r_torch = self._get_ae('torch', ae_type).train(data)
        r_tf = self._get_ae('tf', ae_type).train(data)
        ratio = np.mean(r_torch) / (np.mean(r_tf) + 1e-10)
        assert 0.001 < ratio < 1000
