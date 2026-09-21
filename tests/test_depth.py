import json

import numpy as np
import pytest

from singlepass3d.depth import DepthResult, cached_depth


class FakeEstimator:
    model_id = "test-model"

    def __init__(self):
        self.calls = 0

    def estimate(self, images):
        self.calls += 1
        return [DepthResult(np.ones((2, 3), dtype=np.float32), self.model_id) for _ in images]


def test_depth_cache_and_labels(tmp_path):
    images = [tmp_path / f"{i}.jpg" for i in range(3)]
    for image in images:
        image.write_bytes(b"image")
    estimator = FakeEstimator()
    paths = cached_depth(images, tmp_path / "cache", estimator, batch_size=2)
    assert estimator.calls == 2
    assert cached_depth(images, tmp_path / "cache", estimator) == paths
    assert estimator.calls == 2
    with np.load(paths[0]) as saved:
        assert saved["depth"].shape == (2, 3)
        assert json.loads(str(saved["metadata"]))["geometry_class"] == "inferred"
    with pytest.raises(ValueError):
        DepthResult(np.ones((2, 3)), "test", units="meters")
