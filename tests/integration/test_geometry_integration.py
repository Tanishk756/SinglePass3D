from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from singlepass3d.mesh import generate_mesh
from singlepass3d.pointcloud import process_dense_cloud

o3d = pytest.importorskip("open3d")
pytest.importorskip("trimesh")


def test_metric_pointcloud_and_mesh_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(1800, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = directions * (2.0 + rng.normal(0.0, 0.015, size=(1800, 1)))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector((directions + 1.0) / 2.0)
    source = tmp_path / "dense.ply"
    assert o3d.io.write_point_cloud(str(source), cloud, write_ascii=True)

    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "scale": 1.5,
        "rotation": np.eye(3).tolist(),
        "translation_enu_m": [10.0, 20.0, 30.0],
    }), encoding="utf-8")

    cloud_output = tmp_path / "pointcloud"
    cloud_metrics = process_dense_cloud(
        source, reference, cloud_output, voxel_size_m=0.08, neighbors=10)
    assert cloud_metrics["source_points"] == 1800
    assert cloud_metrics["processed_points"] > 1000
    assert (cloud_output / "raw.ply").is_file()
    assert (cloud_output / "processed.ply").is_file()

    mesh_output = tmp_path / "mesh"
    mesh_metrics = generate_mesh(
        cloud_output / "processed.ply", mesh_output, depth=6, min_points=500)
    assert mesh_metrics["vertices"] > 0
    assert mesh_metrics["triangles"] > 0
    assert mesh_metrics["cropped_to_observed_bounds"] is True
    assert (mesh_output / "scene.obj").stat().st_size > 0
    assert (mesh_output / "scene.glb").stat().st_size > 0
