import numpy as np
import pytest

from singlepass3d.pointcloud import transform_matrix


def test_metric_transform_matrix():
    reference = {"scale": 2, "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                 "translation_enu_m": [10, 20, 30]}
    matrix = transform_matrix(reference)
    point = matrix @ np.array([1, 2, 3, 1])
    assert np.allclose(point[:3], [6, 22, 36])
    with pytest.raises(ValueError):
        transform_matrix({**reference, "scale": -1})
