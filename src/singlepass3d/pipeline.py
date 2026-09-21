"""First-milestone orchestration and factual report generation."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from html import escape
from pathlib import Path

from .colmap import discover_colmap, read_sparse_text, reconstruct_sparse
from .config import PipelineConfig
from .core import ensure_output, file_identifier, run_stage
from .telemetry import load_telemetry, parse_timestamp, synchronize
from .video import extract_frames, inspect_video


def build_report(root: Path) -> Path:
    """Summarize only values present in completed stage artifacts."""
    info = json.loads((root / "video_info.json").read_text(encoding="utf-8"))
    with (root / "frames/frame_manifest.csv").open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    sparse = read_sparse_text(root / "reconstruction/text_model")
    alignment = json.loads((root / "geospatial/metrics.json").read_text(encoding="utf-8"))
    report = {
        "video": info,
        "frame_processing": {
            "considered": len(frames),
            "accepted": sum(row["selected"].lower() in {"true", "1"} for row in frames),
            "blur_rejections": sum(row["rejection_reason"] == "blur" for row in frames),
            "exposure_rejections": sum(row["rejection_reason"] == "exposure" for row in frames),
            "motion_rejections": sum(row["rejection_reason"] == "low_motion" for row in frames),
        },
        "sfm": {
            "registered_images": len(sparse.poses),
            "sparse_points": sparse.point_count,
            "observations": sparse.observations,
        },
        "gps_alignment": alignment,
    }
    destination = ensure_output(root / "reports")
    metrics = destination / "metrics.json"
    metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    page = destination / "report.html"
    pretty = escape(json.dumps(report, indent=2))
    page.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>SinglePass3D processing report</title>"
        "<style>body{font:16px system-ui;max-width:900px;margin:3rem auto;"
        "padding:0 1rem}pre{white-space:pre-wrap;background:#f2f4f6;padding:1rem}</style>"
        "<h1>SinglePass3D processing report</h1>"
        "<p>GPS residuals measure alignment fit, not absolute survey accuracy.</p>"
        f"<pre>{pretty}</pre></html>",
        encoding="utf-8",
    )
    return page


def reconstruct(video: Path, telemetry: Path, output: Path,
                config: PipelineConfig, start_time: str) -> Path:
    """Run inspection, extraction, GPS sync, COLMAP and metric alignment."""
    root = ensure_output(output)
    video_id = file_identifier(video)
    telemetry_id = file_identifier(telemetry)
    config_hash = config.digest()
    def inspect_action() -> list[Path]:
        info = inspect_video(video)
        artifact = root / "video_info.json"
        artifact.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
        return [artifact]
    run_stage(root, "inspect_video", 1, config.digest_for("video"), {"video": video_id}, inspect_action)
    frames = run_stage(
        root, "extract_frames", 1, config.digest_for("video", "frame_selection"), {"video": video_id},
        lambda: [extract_frames(video, root / "frames", config)],
    )[0]
    samples = load_telemetry(telemetry)
    start_seconds = parse_timestamp(start_time)
    def sync_action() -> list[Path]:
        destination = root / "telemetry/frame_telemetry.csv"
        synchronize(frames, samples, start_seconds, destination,
                    config.gps.sync_method, config.gps.tolerance_seconds)
        return [destination]
    synchronized = run_stage(
        root, "sync_telemetry", 1, config.digest_for("gps"),
        {"manifest": file_identifier(frames), "telemetry": telemetry_id,
         "start_time": str(start_seconds)}, sync_action,
    )[0]
    images = sorted((root / "frames").glob("*.jpg"))
    if not images:
        raise ValueError("Frame extraction selected no usable images")
    image_ids = {item.name: file_identifier(item) for item in images}
    def sparse_action() -> list[Path]:
        reconstruct_sparse(root / "frames", root / "reconstruction",
                           discover_colmap(config.colmap.executable), config.colmap.camera_model,
                           config.colmap.use_gpu, config.colmap.sequential_overlap)
        model = root / "reconstruction/text_model"
        return [model / name for name in ("images.txt", "points3D.txt", "cameras.txt")]
    model_files = run_stage(root, "sparse", 1, config.digest_for("colmap"), image_ids, sparse_action)
    def align_action() -> list[Path]:
        from .geoexport import align_sparse
        destination = root / "geospatial"
        align_sparse(model_files[0].parent, synchronized, destination)
        return [destination / name for name in
                ("sparse_georeferenced.ply", "camera_poses.json", "trajectory.geojson", "reference.json", "metrics.json")]
    alignment_files = run_stage(
        root, "align_sparse", 1, config.digest_for("gps"),
        {"images": file_identifier(model_files[0]),
         "points": file_identifier(model_files[1]),
         "synchronized": file_identifier(synchronized)}, align_action,
    )
    report = run_stage(
        root, "report", 1, "report-v1",
        {"alignment": file_identifier(alignment_files[-1]),
         "frames": file_identifier(frames)},
        lambda: [build_report(root), root / "reports/metrics.json"],
    )[0]
    manifest = {
        "video": str(video.resolve()), "telemetry": str(telemetry.resolve()),
        "configuration_hash": config_hash, "report": str(report.relative_to(root)),
        "coordinate_frame": "local ENU meters",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report
