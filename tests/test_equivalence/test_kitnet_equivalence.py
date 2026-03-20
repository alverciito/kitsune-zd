"""Cross-framework KitNET equivalence tests."""
import numpy as np
import pytest

from src.torch.kitnet import KitNET as TorchKitNET
from src.tf.kitnet import KitNET as TFKitNET


def _make_data(n, d, seed=42):
    rng = np.random.RandomState(seed)
    return rng.rand(n, d).astype(np.float32)


class TestKitNETEquivalence:
    """Verify both backends produce comparable KitNET results."""

    @pytest.mark.parametrize("ae_type", ["elm", "stat"])
    def test_numpy_ae_identical_scores(self, ae_type):
        """NumPy-based AEs must produce identical feature maps and scores."""
        data = _make_data(500, 8)
        kn1 = TorchKitNET(8, ae_type=ae_type, clustering='corr')
        kn2 = TFKitNET(8, ae_type=ae_type, clustering='corr')
        s1 = kn1.run(data)
        s2 = kn2.run(data)
        np.testing.assert_allclose(s1, s2, rtol=1e-5)

    def test_both_produce_scores(self):
        """Both backends produce non-empty score arrays."""
        data = _make_data(500, 8)
        for Backend in [TorchKitNET, TFKitNET]:
            kn = Backend(8, ae_type='elm', clustering='corr')
            scores = kn.run(data)
            assert len(scores) > 0
            assert np.all(np.isfinite(scores))
