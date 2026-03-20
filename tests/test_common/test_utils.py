"""Tests for src/common/utils.py: windowing, normalization, RMSE."""
import numpy as np
import pytest
from src.common.utils import create_windows, create_windows_ar, normalize_minmax, compute_rmse_per_sample, sigmoid
from src.common.config import EPSILON


class TestCreateWindows:
    """Sliding window creation tests."""

    def test_basic_shape(self):
        x = np.random.rand(100, 5).astype(np.float32)
        w = create_windows(x, 10)
        assert w.shape == (91, 10, 5)

    def test_minimum_input(self):
        """Exact window_size samples -> 1 window."""
        x = np.random.rand(10, 3).astype(np.float32)
        w = create_windows(x, 10)
        assert w.shape == (1, 10, 3)
        np.testing.assert_array_equal(w[0], x)

    def test_window_content_correctness(self):
        """Each window[i] must be x[i:i+window_size]."""
        x = np.arange(20).reshape(10, 2).astype(np.float32)
        w = create_windows(x, 3)
        for i in range(len(w)):
            np.testing.assert_array_equal(w[i], x[i:i+3])

    def test_too_short_raises(self):
        """Input shorter than window_size must raise."""
        x = np.random.rand(5, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Input length 5 < window_size 10"):
            create_windows(x, 10)

    def test_single_feature(self):
        """Works with 1 feature column."""
        x = np.random.rand(50, 1).astype(np.float32)
        w = create_windows(x, 5)
        assert w.shape == (46, 5, 1)

    def test_large_window_equals_input(self):
        """window_size == N -> single window == input."""
        x = np.random.rand(20, 4).astype(np.float32)
        w = create_windows(x, 20)
        assert w.shape == (1, 20, 4)
        np.testing.assert_array_equal(w[0], x)

    def test_no_data_leakage_between_windows(self):
        """Window i should not contain data from beyond x[i+window_size-1]."""
        x = np.zeros((20, 2), dtype=np.float32)
        x[10:, :] = 999.0  # mark second half
        w = create_windows(x, 5)
        # Window 5 covers x[5:10] — should be all zeros
        assert np.all(w[5] == 0.0)
        # Window 6 covers x[6:11] — last row has 999
        assert w[6, -1, 0] == 999.0
        assert np.all(w[6, :-1, :] == 0.0)


class TestCreateWindowsAR:
    """Autoregressive windowing tests."""

    def test_basic_shapes(self):
        x = np.random.rand(20, 3).astype(np.float32)
        inputs, targets = create_windows_ar(x, 5)
        assert inputs.shape == (16, 4, 3)  # window_size-1 = 4
        assert targets.shape == (16, 3)

    def test_target_is_last_frame(self):
        """Target[i] must equal x[i + window_size - 1]."""
        x = np.arange(30).reshape(15, 2).astype(np.float32)
        inputs, targets = create_windows_ar(x, 5)
        for i in range(len(targets)):
            np.testing.assert_array_equal(targets[i], x[i + 4])

    def test_input_is_prefix(self):
        """Input[i] must equal x[i:i+window_size-1]."""
        x = np.arange(30).reshape(15, 2).astype(np.float32)
        inputs, targets = create_windows_ar(x, 5)
        for i in range(len(inputs)):
            np.testing.assert_array_equal(inputs[i], x[i:i+4])

    def test_minimum_input(self):
        """Exact window_size -> 1 window pair."""
        x = np.random.rand(5, 2).astype(np.float32)
        inputs, targets = create_windows_ar(x, 5)
        assert inputs.shape == (1, 4, 2)
        assert targets.shape == (1, 2)


class TestNormalize:
    """Min-max normalization tests."""

    def test_output_range(self):
        x = np.array([[1, 10], [5, 50], [9, 90]], dtype=np.float32)
        norm_min = np.array([1, 10], dtype=np.float32)
        norm_max = np.array([9, 90], dtype=np.float32)
        result = normalize_minmax(x, norm_min, norm_max)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_constant_column(self):
        """Constant column -> all ~0 (not NaN/Inf)."""
        x = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float32)
        norm_min = x.min(axis=0)
        norm_max = x.max(axis=0)
        result = normalize_minmax(x, norm_min, norm_max)
        assert np.all(np.isfinite(result))

    def test_single_sample(self):
        """Single sample -> all ~0 (division by epsilon)."""
        x = np.array([[3.0, 7.0]])
        result = normalize_minmax(x, x[0], x[0])
        assert np.all(np.isfinite(result))


class TestRMSE:
    """RMSE computation tests."""

    def test_identical_is_zero(self):
        x = np.random.rand(10, 5)
        rmse = compute_rmse_per_sample(x, x)
        np.testing.assert_allclose(rmse, 0.0, atol=1e-10)

    def test_known_value(self):
        x = np.array([[0.0, 0.0]])
        z = np.array([[3.0, 4.0]])
        rmse = compute_rmse_per_sample(x, z)
        # sqrt(mean([9, 16])) = sqrt(12.5) ≈ 3.5355
        np.testing.assert_allclose(rmse, np.sqrt(12.5), rtol=1e-6)

    def test_batch_independence(self):
        """Each sample's RMSE is independent."""
        x = np.zeros((3, 2))
        z = np.array([[1, 0], [0, 0], [0, 2]], dtype=float)
        rmse = compute_rmse_per_sample(x, z)
        np.testing.assert_allclose(rmse[0], np.sqrt(0.5))
        np.testing.assert_allclose(rmse[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(rmse[2], np.sqrt(2.0))


class TestSigmoid:
    """Numerically stable sigmoid tests."""

    def test_zero(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive(self):
        result = sigmoid(np.array([100.0]))[0]
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self):
        result = sigmoid(np.array([-100.0]))[0]
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_no_overflow(self):
        """Must not produce Inf/NaN for extreme values."""
        extremes = np.array([-1000, -500, -100, 100, 500, 1000], dtype=np.float64)
        result = sigmoid(extremes)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_symmetry(self):
        x = np.linspace(-5, 5, 100)
        np.testing.assert_allclose(sigmoid(x) + sigmoid(-x), 1.0, atol=1e-10)
