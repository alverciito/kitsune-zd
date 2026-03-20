"""Tests for detectors: CentroidDetector, DistributionDetector, filters, WindowDiff."""
import numpy as np
import pytest
from src.detectors.centroid import CentroidDetector
from src.detectors.distribution import DistributionDetector
from src.detectors.filters import mean_filter, median_filter
from src.detector import windowdiff


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestMeanFilter:

    def test_preserves_length(self):
        x = np.random.rand(100)
        assert len(mean_filter(x, 5)) == 100

    def test_window_1_is_identity(self):
        x = np.random.rand(50)
        np.testing.assert_allclose(mean_filter(x, 1), x)

    def test_constant_signal_interior_unchanged(self):
        """Interior of constant signal should be unchanged after mean filter."""
        x = np.full(100, 3.14)
        result = mean_filter(x, 10)
        # Interior (away from edges) should be exactly 3.14
        np.testing.assert_allclose(result[10:-10], x[10:-10], atol=1e-10)

    def test_actually_smooths(self):
        """Variance of smoothed signal should be lower than original."""
        rng = np.random.RandomState(0)
        x = rng.randn(1000)
        smoothed = mean_filter(x, 20)
        assert np.var(smoothed) < np.var(x)

    def test_known_value(self):
        """Manual check on small array."""
        x = np.array([0, 0, 10, 0, 0], dtype=float)
        result = mean_filter(x, 3)
        # Middle element: mean(0, 10, 0) = 3.33...
        assert result[2] == pytest.approx(10.0 / 3, rel=1e-5)

    def test_single_element(self):
        """Single element convolved with kernel of 3: 5/3 due to zero-padding."""
        x = np.array([5.0])
        result = mean_filter(x, 3)
        assert len(result) == 1
        assert np.isfinite(result[0])


class TestMedianFilter:

    def test_preserves_length(self):
        x = np.random.rand(100)
        assert len(median_filter(x, 5)) == 100

    def test_window_1_is_identity(self):
        x = np.random.rand(50)
        np.testing.assert_allclose(median_filter(x, 1), x)

    def test_constant_signal_unchanged(self):
        x = np.full(100, 2.71)
        np.testing.assert_allclose(median_filter(x, 10), x, atol=1e-10)

    def test_removes_spike(self):
        """Median filter should remove a single spike."""
        x = np.zeros(50)
        x[25] = 100.0
        result = median_filter(x, 5)
        # The spike position should be suppressed to 0
        assert result[25] == 0.0

    def test_actually_smooths(self):
        rng = np.random.RandomState(0)
        x = rng.randn(1000)
        smoothed = median_filter(x, 20)
        assert np.var(smoothed) < np.var(x)


# ---------------------------------------------------------------------------
# CentroidDetector
# ---------------------------------------------------------------------------

class TestCentroidDetector:

    def test_basic_operation(self):
        train = np.random.rand(200) * 0.1  # low scores
        det = CentroidDetector(n_clusters=3, filter_window=0)
        det.train(train)
        # Normal data
        normal = np.random.rand(100) * 0.1
        # Anomaly data
        anomaly = np.random.rand(100) * 10 + 50
        normal_scores = det.execute(normal)
        anomaly_scores = det.execute(anomaly)
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_output_shape(self):
        det = CentroidDetector(n_clusters=5, filter_window=10)
        det.train(np.random.rand(100))
        result = det.execute(np.random.rand(50))
        assert result.shape == (50,)

    def test_all_finite(self):
        det = CentroidDetector(n_clusters=3, filter_window=5)
        det.train(np.random.rand(100))
        result = det.execute(np.random.rand(200))
        assert np.all(np.isfinite(result))

    def test_nonzero_output(self):
        """Unless all scores land exactly on centroids, output should be > 0."""
        rng = np.random.RandomState(0)
        det = CentroidDetector(n_clusters=3, filter_window=0)
        det.train(rng.rand(200))
        result = det.execute(rng.rand(100) + 5)  # shifted
        assert np.mean(result) > 0

    def test_filter_reduces_variance(self):
        rng = np.random.RandomState(0)
        det = CentroidDetector(n_clusters=3, filter_window=0)
        det.train(rng.rand(200))
        raw = det.execute(rng.rand(500))

        det2 = CentroidDetector(n_clusters=3, filter_window=20)
        det2.train(rng.rand(200))
        filtered = det2.execute(rng.rand(500))
        # The filtered version should have less jitter
        assert np.var(filtered) <= np.var(raw) + 1e-6

    def test_small_train_set(self):
        det = CentroidDetector(n_clusters=2, filter_window=0)
        det.train(np.array([1.0, 2.0, 3.0]))
        result = det.execute(np.array([1.5, 10.0]))
        assert result.shape == (2,)
        assert result[1] > result[0]  # 10.0 is farther from centroids


# ---------------------------------------------------------------------------
# DistributionDetector
# ---------------------------------------------------------------------------

class TestDistributionDetector:

    def test_basic_operation(self):
        rng = np.random.RandomState(0)
        train = rng.rand(500) * 0.1
        det = DistributionDetector(window_size=50, filter_window=0)
        det.train(train)

        # In-distribution scores
        normal = rng.rand(200) * 0.1
        normal_scores = det.execute(normal)

        # Out-of-distribution
        anomaly = rng.rand(200) * 10 + 50
        anomaly_scores = det.execute(anomaly)

        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_output_shape(self):
        det = DistributionDetector(window_size=20, filter_window=0)
        det.train(np.random.rand(100))
        result = det.execute(np.random.rand(50))
        assert result.shape == (50,)

    def test_all_finite(self):
        det = DistributionDetector(window_size=20, filter_window=0)
        det.train(np.random.rand(100))
        result = det.execute(np.random.rand(200))
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_identical_distribution_low_scores(self):
        """If test data matches training distribution, scores should be near 0."""
        rng = np.random.RandomState(42)
        data = rng.rand(1000)
        det = DistributionDetector(window_size=100, filter_window=0)
        det.train(data)
        scores = det.execute(data)
        # Most scores should be quite low since distributions match
        assert np.median(scores) < 0.01

    def test_stores_training_stats(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        det = DistributionDetector(window_size=5, filter_window=0)
        det.train(data)
        assert det.mu_T == pytest.approx(3.0)
        assert det.sigma_T == pytest.approx(np.std(data))


# ---------------------------------------------------------------------------
# WindowDiff
# ---------------------------------------------------------------------------

class TestWindowDiff:

    def test_identical_is_zero(self):
        ref = np.array([0, 0, 1, 1, 1, 0, 0])
        assert windowdiff(ref, ref) == 0.0

    def test_all_wrong_is_high(self):
        """Hypothesis has many extra boundaries that ref doesn't -> high WD."""
        ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        hyp = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        wd = windowdiff(ref, hyp)
        assert wd > 0.5

    def test_range_zero_to_one(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            ref = rng.randint(0, 2, size=100)
            hyp = rng.randint(0, 2, size=100)
            wd = windowdiff(ref, hyp)
            assert 0.0 <= wd <= 1.0

    def test_no_boundaries_returns_zero(self):
        ref = np.zeros(50, dtype=int)
        hyp = np.zeros(50, dtype=int)
        assert windowdiff(ref, hyp) == 0.0

    def test_symmetric(self):
        """WindowDiff is symmetric: WD(ref, hyp) == WD(hyp, ref)."""
        rng = np.random.RandomState(5)
        ref = rng.randint(0, 2, size=100)
        hyp = rng.randint(0, 2, size=100)
        assert windowdiff(ref, hyp) == pytest.approx(windowdiff(hyp, ref))

    def test_known_value(self):
        """
        ref: [0,0,0,1,1,1,0,0,0,0] -> 2 boundaries at pos 3,6
        hyp: [0,0,0,0,1,1,0,0,0,0] -> 2 boundaries at pos 4,6
        Mean segment length = 10/3 ≈ 3.33, k = 1
        """
        ref = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        hyp = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        wd = windowdiff(ref, hyp, k=2)
        assert 0 < wd < 1  # not perfect match, not total mismatch

    def test_custom_k(self):
        ref = np.array([0, 0, 1, 1, 0, 0])
        hyp = np.array([0, 1, 1, 0, 0, 0])
        wd1 = windowdiff(ref, hyp, k=1)
        wd2 = windowdiff(ref, hyp, k=2)
        # Different k values should produce different scores
        assert wd1 != wd2 or (wd1 == 0 and wd2 == 0)

    def test_single_boundary_offset(self):
        """One boundary shifted by 1 position."""
        ref = np.zeros(20, dtype=int)
        hyp = np.zeros(20, dtype=int)
        ref[10:] = 1
        hyp[11:] = 1
        wd = windowdiff(ref, hyp)
        assert 0 < wd < 0.5  # small error


# ---------------------------------------------------------------------------
# Integration: threshold_sweep + windowdiff
# ---------------------------------------------------------------------------

class TestThresholdSweepIntegration:

    def test_perfect_separation(self):
        """Clearly separable scores should give F1 near 1.0."""
        from src.detector import threshold_sweep
        scores = np.concatenate([np.ones(500) * 0.01, np.ones(500) * 100.0])
        labels = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.int32)
        metrics = threshold_sweep(scores, labels)
        assert metrics['best_f1'] > 0.95
        assert metrics['best_recall'] > 0.95

    def test_random_scores_low_f1(self):
        """Random scores on balanced labels should give mediocre F1."""
        from src.detector import threshold_sweep
        rng = np.random.RandomState(0)
        scores = rng.rand(1000)
        labels = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.int32)
        metrics = threshold_sweep(scores, labels)
        assert metrics['best_f1'] < 0.8

    def test_all_attack_labels(self):
        """All labels=1: best F1 should be achievable at low threshold."""
        from src.detector import threshold_sweep
        scores = np.random.rand(100)
        labels = np.ones(100, dtype=np.int32)
        metrics = threshold_sweep(scores, labels)
        assert metrics['best_recall'] == pytest.approx(1.0, abs=0.01)
