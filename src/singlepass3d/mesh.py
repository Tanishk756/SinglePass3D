"""Optional surface interpolation from metric dense point clouds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import ensure_output


def generate_mesh(source: Path, output: Path, depth: int = 8,
                  min_points: int = 1000, density_quantile: float = 0.05) -> dict:
    """Poisson surface is inferred between observed points; no texture is claimed."""
    if depth < 5 or depth > 12 or min_points < 3 or not 0 <= density_quantile < 1:
        raise ValueError("Invalid meshing parameters")
    try:
        import open3d as o3d
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Meshing requires pip install '.[pointcloud,mesh]'") from exc
    cloud = o3d.io.read_point_cloud(str(source))
    count = len(cloud.points)
    if count < min_points:
        raise ValueError(f"Only {count} points; meshing requires at least {min_points}")
    cloud.estimate_normals()
    cloud.orient_normals_consistent_tangent_plane(min(30, count - 1))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth)
    cutoff = float(np.quantile(np.asarray(densities), density_quantile))
    mesh.remove_vertices_by_mask(np.asarray(densities) < cutoff)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise ValueError("Meshing produced no valid triangles")
    mesh.compute_vertex_normals()
    output = ensure_output(output)
    obj = output / "scene.obj"
    if not o3d.io.write_triangle_mesh(str(obj), mesh):
        raise OSError(f"Failed to write {obj}")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    vertex_colors = None
    if len(colors) == len(vertices):
        vertex_colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    glb = output / "scene.glb"
    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors,
                    process=False).export(glb)
    metrics = {
        "source_points": count,
        "vertices": len(vertices),
        "triangles": len(faces),
        "coordinate_frame": "local ENU meters",
        "geometry_class": "inferred surface from observed dense points",
        "texture": "vertex colors" if vertex_colors is not None else None,
        "method": "screened Poisson",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
