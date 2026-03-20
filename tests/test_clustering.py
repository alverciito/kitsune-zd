"""Tests for clustering: CorClust, DBSCAN, KMeans."""
import numpy as np
import pytest
from src.clustering import get_clustering
from src.clustering.corclust import CorClust
from src.clustering.dbscan_clust import DBSCANClust
from src.clustering.kmeans_clust import KMeansClust


def _feed_clusterer(clusterer, n_samples=200, n_features=20, seed=42):
    """Feed structured data to a clusterer. Features in groups with correlation."""
    rng = np.random.RandomState(seed)
    # Create 4 correlated groups of ~5 features each
    base_signals = rng.randn(n_samples, 4)
    data = np.zeros((n_samples, n_features))
    for i in range(n_features):
        group = i % 4
        data[:, i] = base_signals[:, group] + rng.randn(n_samples) * 0.2
    for i in range(n_samples):
        clusterer.update(data[i])
    return data


class TestCorClust:

    def test_produces_valid_feature_map(self):
        c = CorClust(20)
        _feed_clusterer(c)
        fmap = c.cluster(10)
        assert isinstance(fmap, list)
        assert len(fmap) > 0
        # Every feature appears exactly once
        all_features = sorted([f for group in fmap for f in group])
        assert all_features == list(range(20))

    def test_max_cluster_size_respected(self):
        c = CorClust(20)
        _feed_clusterer(c)
        fmap = c.cluster(5)
        for group in fmap:
            assert len(group) <= 5, f"Cluster too big: {len(group)} > 5"

    def test_single_feature(self):
        c = CorClust(1)
        for _ in range(50):
            c.update(np.array([np.random.rand()]))
        fmap = c.cluster(1)
        assert fmap == [[0]]

    def test_two_features(self):
        c = CorClust(2)
        for _ in range(50):
            x = np.random.rand(2)
            c.update(x)
        fmap = c.cluster(2)
        all_f = sorted([f for g in fmap for f in g])
        assert all_f == [0, 1]

    def test_no_empty_clusters(self):
        c = CorClust(15)
        _feed_clusterer(c, n_features=15)
        fmap = c.cluster(5)
        for group in fmap:
            assert len(group) >= 1

    def test_corr_dist_shape(self):
        c = CorClust(10)
        _feed_clusterer(c, n_features=10, n_samples=100)
        D = c.corr_dist()
        assert D.shape == (10, 10)
        assert np.all(np.isfinite(D))
        assert np.all(D >= 0)


class TestDBSCANClust:

    def test_produces_valid_feature_map(self):
        c = DBSCANClust(20)
        _feed_clusterer(c)
        fmap = c.cluster(5)
        assert isinstance(fmap, list)
        assert len(fmap) > 0
        # All features assigned (DBSCAN may merge some)
        all_features = sorted([f for g in fmap for f in g])
        # Must contain all features 0..19
        assert set(all_features) == set(range(20))

    def test_no_empty_clusters(self):
        c = DBSCANClust(20)
        _feed_clusterer(c)
        fmap = c.cluster(5)
        for g in fmap:
            assert len(g) >= 1

    def test_small_feature_count(self):
        c = DBSCANClust(4)
        rng = np.random.RandomState(7)
        for _ in range(100):
            c.update(rng.randn(4))
        fmap = c.cluster(2)
        all_f = sorted([f for g in fmap for f in g])
        assert set(all_f) == set(range(4))


class TestKMeansClust:

    def test_produces_valid_feature_map(self):
        c = KMeansClust(20)
        _feed_clusterer(c)
        fmap = c.cluster(5)
        assert isinstance(fmap, list)
        assert len(fmap) == 5
        all_features = sorted([f for g in fmap for f in g])
        assert all_features == list(range(20))

    def test_no_empty_clusters(self):
        c = KMeansClust(20)
        _feed_clusterer(c)
        fmap = c.cluster(5)
        for g in fmap:
            assert len(g) >= 1

    def test_deterministic(self):
        """Same data -> same result."""
        fmaps = []
        for _ in range(2):
            c = KMeansClust(10)
            rng = np.random.RandomState(123)
            for _ in range(100):
                c.update(rng.randn(10))
            fmaps.append(c.cluster(3))
        for g1, g2 in zip(fmaps[0], fmaps[1]):
            assert sorted(g1) == sorted(g2)


class TestClusteringFactory:

    def test_corr(self):
        c = get_clustering('corr', 10)
        assert isinstance(c, CorClust)

    def test_dbscan(self):
        c = get_clustering('dbscan', 10)
        assert isinstance(c, DBSCANClust)

    def test_kmeans(self):
        c = get_clustering('kmeans', 10)
        assert isinstance(c, KMeansClust)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_clustering('spectral', 10)


class TestClusteringEdgeCases:

    @pytest.mark.parametrize("clust_class", [CorClust, DBSCANClust, KMeansClust])
    def test_constant_data(self, clust_class):
        """All-constant data should not crash."""
        c = clust_class(5)
        for _ in range(100):
            c.update(np.ones(5))
        fmap = c.cluster(2)
        assert isinstance(fmap, list)
        all_f = set(f for g in fmap for f in g)
        assert all_f == set(range(5))

    @pytest.mark.parametrize("clust_class", [CorClust, KMeansClust])
    def test_request_more_clusters_than_features(self, clust_class):
        """Requesting more clusters than features should not crash."""
        c = clust_class(5)
        rng = np.random.RandomState(0)
        for _ in range(100):
            c.update(rng.randn(5))
        fmap = c.cluster(20)
        all_f = set(f for g in fmap for f in g)
        assert all_f == set(range(5))
