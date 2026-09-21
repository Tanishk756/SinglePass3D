"""Georeference COLMAP sparse points and camera positions from synchronized GPS."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .colmap import read_sparse_text
from .core import ensure_output
from .geospatial import enu_to_wgs84, robust_alignment, wgs84_to_enu


def write_ply(source: Path, output: Path, transform) -> int:
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            values = line.split()
            if len(values) < 8:
                raise ValueError("Malformed COLMAP points3D.txt row")
            position = np.array([float(value) for value in values[1:4]])
            color = [int(value) for value in values[4:7]]
            rows.append((*transform(position.reshape(1, 3))[0], *color))
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(rows)}\n")
        for property_type, name in [
            ("float", "x"), ("float", "y"), ("float", "z"),
            ("uchar", "red"), ("uchar", "green"), ("uchar", "blue")
        ]:
            stream.write(f"property {property_type} {name}\n")
        stream.write("end_header\n")
        for row in rows:
            stream.write(" ".join(str(value) for value in row) + "\n")
    return len(rows)


def align_sparse(model: Path, synchronized: Path, output: Path, threshold_m: float = 5.0) -> dict:
    """Align registered visual cameras to matched GPS and write metric artifacts."""
    result = read_sparse_text(model)
    with synchronized.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    gps_by_name = {row["filename"]: row for row in rows if row["matched"].lower() in {"true", "1"}}
    pairs = [(pose, gps_by_name[pose.name]) for pose in result.poses if pose.name in gps_by_name]
    if len(pairs) < 3:
        raise ValueError("At least three registered cameras with synchronized GPS are required")
    origin = tuple(float(pairs[0][1][key]) for key in ("latitude", "longitude", "altitude"))
    coordinates = np.array([
        [float(row[key]) for key in ("latitude", "longitude", "altitude")]
        for _, row in pairs
    ])
    gps_enu = wgs84_to_enu(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], origin)
    visual = np.array([pose.center for pose, _ in pairs])
    alignment = robust_alignment(visual, gps_enu, threshold_m=threshold_m)
    output = ensure_output(output)
    cloud = output / "sparse_georeferenced.ply"
    count = write_ply(model / "points3D.txt", cloud, alignment.transform)
    trajectory = [
        {"image": pose.name, "visual_center": pose.center,
         "enu_m": aligned.tolist(),
         "gps_enu_m": gps.tolist(),
         "gps_wgs84": [float(row["longitude"]), float(row["latitude"]), float(row["altitude"])],
         "residual_m": float(residual), "alignment_inlier": bool(inlier)}
        for (pose, row), aligned, gps, residual, inlier in zip(
            pairs, alignment.transform(visual), gps_enu, alignment.residuals, alignment.inliers
        )
    ]
    visual_wgs84 = enu_to_wgs84(alignment.transform(visual), origin)
    trajectory_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"trajectory": "visual_aligned"},
             "geometry": {"type": "LineString", "coordinates": visual_wgs84.tolist()}},
            {"type": "Feature", "properties": {"trajectory": "gps_supplied"},
             "geometry": {"type": "LineString", "coordinates": [
                 [float(row["longitude"]), float(row["latitude"]), float(row["altitude"])]
                 for _, row in pairs]}},
        ],
    }
    (output / "trajectory.geojson").write_text(
        json.dumps(trajectory_geojson, indent=2), encoding="utf-8")
    metrics = alignment.metrics()
    metrics.update({"registered_images": len(result.poses), "sparse_points": count,
                    "observations": result.observations})
    reference = {
        "coordinate_frame": "local ENU meters",
        "altitude_assumption": "Input altitude treated as WGS84 ellipsoidal meters",
        "origin_wgs84": {"latitude": origin[0], "longitude": origin[1], "altitude": origin[2]},
        "scale": alignment.scale,
        "rotation": alignment.rotation.tolist(),
        "translation_enu_m": alignment.translation.tolist(),
        "note": "Observed COLMAP geometry aligned to GPS; residuals are alignment residuals, not absolute accuracy",
    }
    (output / "camera_poses.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    (output / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
