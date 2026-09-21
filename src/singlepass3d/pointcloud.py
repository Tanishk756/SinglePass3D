"""Optional observed dense point-cloud processing in the metric ENU frame."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import ensure_output


def transform_matrix(reference: dict) -> np.ndarray:
    scale = float(reference["scale"])
    rotation = np.asarray(reference["rotation"], dtype=float)
    translation = np.asarray(reference["translation_enu_m"], dtype=float)
    if scale <= 0 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Invalid similarity transform in reference file")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Reference rotation is not orthonormal")
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def process_dense_cloud(source: Path, reference_path: Path, output: Path,
                        voxel_size_m: float = 0.1, neighbors: int = 20,
                        std_ratio: float = 2.0) -> dict:
    """Keep raw georeferenced points and a separate filtered cloud."""
    if voxel_size_m <= 0 or neighbors < 2 or std_ratio <= 0:
        raise ValueError("Voxel size, neighbor count and std ratio must be positive")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Point-cloud processing requires pip install '.[pointcloud]'") from exc
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    cloud = o3d.io.read_point_cloud(str(source))
    source_count = len(cloud.points)
    if source_count == 0:
        raise ValueError("Dense PLY contains no points")
    cloud.transform(transform_matrix(reference))
    output = ensure_output(output)
    raw = output / "raw.ply"
    if not o3d.io.write_point_cloud(str(raw), cloud):
        raise OSError(f"Failed to write {raw}")
    processed = cloud.voxel_down_sample(voxel_size_m)
    if len(processed.points) >= neighbors:
        processed, _ = processed.remove_statistical_outlier(
            nb_neighbors=neighbors, std_ratio=std_ratio)
    if len(processed.points) == 0:
        raise ValueError("Filtering removed every point; relax thresholds")
    filtered = output / "processed.ply"
    if not o3d.io.write_point_cloud(str(filtered), processed):
        raise OSError(f"Failed to write {filtered}")
    bounds = processed.get_axis_aligned_bounding_box()
    metrics = {
        "source_points": source_count,
        "processed_points": len(processed.points),
        "min_enu_m": np.asarray(bounds.min_bound).tolist(),
        "max_enu_m": np.asarray(bounds.max_bound).tolist(),
        "coordinate_frame": "local ENU meters",
        "geometry_class": "observed",
        "method": "COLMAP stereo fusion, voxel downsampling, statistical filtering",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
