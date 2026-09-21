import pytest

from singlepass3d.mesh import generate_mesh


def test_invalid_mesh_parameters(tmp_path):
    with pytest.raises(ValueError):
        generate_mesh(tmp_path / "missing.ply", tmp_path, depth=4)
    with pytest.raises(ValueError):
        generate_mesh(tmp_path / "missing.ply", tmp_path, density_quantile=1)
