"""End-to-end tests for KitNET pipeline (TensorFlow backend)."""
import numpy as np
import pytest
from src.tf.kitnet import KitNET


def _synth_dataset(n_samples=1500, n_features=20, attack_start=1200, seed=42):
    """
    Synthetic dataset: benign = low-amplitude sine waves, attack = high-amplitude random.
    With n_samples=1500, fm_grace=500, ad_grace=500 -> 500 execution samples.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10 * np.pi, n_samples)
    X = np.column_stack([np.sin(t + i) + rng.randn(n_samples) * 0.05
                         for i in range(n_features)]).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.int32)
    # Inject anomaly
    X[attack_start:] = rng.randn(n_samples - attack_start, n_features) * 20 + 100
    y[attack_start:] = 1
    return X, y


class TestKitNETEndToEnd:
    """Full pipeline tests — actually runs the 3-phase pipeline."""

    def test_elm_produces_scores(self):
        X, y = _synth_dataset()
        kn = KitNET(n_features=20, ae_type='elm', fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        assert len(scores) == 500
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)
        assert np.mean(scores) > 0

    def test_elm_detects_anomaly(self):
        """Attack portion should have higher scores than benign."""
        X, y = _synth_dataset(n_samples=1500, attack_start=1200)
        kn = KitNET(n_features=20, ae_type='elm', fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        benign_scores = scores[:200]
        attack_scores = scores[200:]
        assert np.mean(attack_scores) > np.mean(benign_scores), \
            f"Attack mean {np.mean(attack_scores):.4f} should be > benign {np.mean(benign_scores):.4f}"

    def test_stat_produces_scores(self):
        X, y = _synth_dataset()
        kn = KitNET(n_features=20, ae_type='stat', fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        assert len(scores) == 500
        assert np.all(np.isfinite(scores))
        assert np.mean(scores) > 0

    def test_stat_detects_anomaly(self):
        X, y = _synth_dataset(n_samples=1500, attack_start=1200)
        kn = KitNET(n_features=20, ae_type='stat', fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        benign = scores[:200]
        attack = scores[200:]
        assert np.mean(attack) > np.mean(benign)

    @pytest.mark.parametrize("ae_type", ['conv1d', 'conv2d', 'transformer', 'deep_mlp'])
    def test_dl_produces_scores(self, ae_type):
        X, y = _synth_dataset(n_samples=1200, n_features=10)
        kn = KitNET(n_features=10, ae_type=ae_type, fm_grace=400, ad_grace=400,
                     seq_len=10, exec_window=100, device='cpu')
        scores = kn.run(X)
        assert len(scores) > 0
        assert np.all(np.isfinite(scores))
        assert np.mean(scores) > 0

    def test_no_execution_data_returns_empty(self):
        """If all data is consumed by grace periods, return empty."""
        X = np.random.rand(100, 5).astype(np.float32)
        kn = KitNET(n_features=5, ae_type='elm', fm_grace=50, ad_grace=50)
        scores = kn.run(X)
        assert len(scores) == 0


class TestKitNETClustering:
    """Test different clustering methods in the pipeline."""

    @pytest.mark.parametrize("method", ['corr', 'dbscan', 'kmeans'])
    def test_clustering_method(self, method):
        X, y = _synth_dataset()
        kn = KitNET(n_features=20, ae_type='elm', clustering=method,
                     fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        assert len(scores) == 500
        assert np.all(np.isfinite(scores))
        assert np.mean(scores) > 0

    def test_feature_map_covers_all_features(self):
        """Every feature index must appear in exactly one cluster."""
        X, _ = _synth_dataset()
        kn = KitNET(n_features=20, ae_type='elm', fm_grace=500, ad_grace=500)
        for i in range(500):
            kn.clusterer.update(X[i])
        fmap = kn.clusterer.cluster(kn.max_ae_size)
        all_f = sorted([f for g in fmap for f in g])
        assert all_f == list(range(20))


class TestKitNETOutputAE:
    """Test ELM vs Statistical output layer."""

    @pytest.mark.parametrize("output_type", ['elm', 'stat'])
    def test_output_ae_type(self, output_type):
        X, y = _synth_dataset()
        kn = KitNET(n_features=20, ae_type='elm', output_ae_type=output_type,
                     fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        assert len(scores) == 500
        assert np.all(np.isfinite(scores))
        assert np.mean(scores) > 0


class TestKitNETEdgeCases:

    def test_single_feature(self):
        """Pipeline with 1 feature."""
        rng = np.random.RandomState(0)
        X = rng.randn(1500, 1).astype(np.float32)
        kn = KitNET(n_features=1, ae_type='elm', fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        assert len(scores) == 500
        assert np.all(np.isfinite(scores))

    def test_two_features(self):
        rng = np.random.RandomState(0)
        X = rng.randn(1500, 2).astype(np.float32)
        kn = KitNET(n_features=2, ae_type='elm', fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        assert len(scores) == 500

    def test_max_ae_size_1(self):
        """Each feature gets its own AE."""
        X, _ = _synth_dataset(n_features=5)
        kn = KitNET(n_features=5, ae_type='elm', max_ae_size=1,
                     clustering='corr', fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        assert len(scores) == 500
        assert len(kn.ensemble) == 5

    def test_constant_features_no_crash(self):
        """All-constant input should not crash."""
        X = np.full((1500, 5), 3.0, dtype=np.float32)
        kn = KitNET(n_features=5, ae_type='elm', fm_grace=500, ad_grace=500)
        scores = kn.run(X)
        assert len(scores) == 500
        assert np.all(np.isfinite(scores))

    def test_dl_with_small_exec_window(self):
        """exec_window < number of execution samples: verify batching works."""
        X, _ = _synth_dataset(n_samples=1200, n_features=6)
        kn = KitNET(n_features=6, ae_type='conv1d', fm_grace=400, ad_grace=400,
                     seq_len=10, exec_window=50, device='cpu')
        scores = kn.run(X)
        assert len(scores) > 0
        assert np.all(np.isfinite(scores))
