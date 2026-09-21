import numpy as np
import pytest

from singlepass3d.geospatial import robust_alignment, similarity


def test_similarity_and_outlier():
    rng = np.random.default_rng(42)
    source = rng.normal(size=(20, 3))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = 2.5 * (source @ rotation.T) + [10, 20, 30]
    target[-1] += 100
    aligned = robust_alignment(source, target, threshold_m=0.01)
    assert aligned.inliers.sum() == 19
    assert aligned.scale == pytest.approx(2.5)
    assert aligned.metrics()["rmse_m"] < 1e-10
    assert np.allclose(aligned.transform(source[:-1]), target[:-1])


def test_degenerate_positions():
    with pytest.raises(ValueError):
        similarity(np.zeros((3, 3)), np.ones((3, 3)))
