"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .colmap import discover_colmap, read_sparse_text, reconstruct_sparse
from .config import load_config
from .core import ensure_output, file_identifier, run_stage
from .doctor import doctor, doctor_exit
from .mvs import ply_vertex_count, reconstruct_dense
from .pipeline import build_report, reconstruct, reconstruct_full
from .telemetry import load_telemetry, parse_timestamp, synchronize
from .video import extract_frames, inspect_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="singlepass3d")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor")
    check.add_argument("--output", type=Path, default=Path("data/output"))
    inspect = commands.add_parser("inspect-video")
    inspect.add_argument("video", type=Path)
    extract = commands.add_parser("extract-frames")
    extract.add_argument("video", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--config", type=Path)
    telemetry = commands.add_parser("telemetry")
    telemetry.add_argument("path", type=Path)
    sync = commands.add_parser("sync-telemetry")
    sync.add_argument("--manifest", type=Path, required=True)
    sync.add_argument("--telemetry", type=Path, required=True)
    sync.add_argument("--start-time", required=True,
                      help="UTC time of video frame zero (ISO-8601 or Unix seconds)")
    sync.add_argument("--output", type=Path, required=True)
    sync.add_argument("--method", choices=["nearest", "interpolate"], default="interpolate")
    sync.add_argument("--tolerance", type=float, default=1.0)
    sparse = commands.add_parser("reconstruct-sparse")
    sparse.add_argument("--images", type=Path, required=True)
    sparse.add_argument("--output", type=Path, required=True)
    sparse.add_argument("--config", type=Path)
    align = commands.add_parser("align-sparse")
    align.add_argument("--model", type=Path, required=True)
    align.add_argument("--synchronized", type=Path, required=True)
    align.add_argument("--output", type=Path, required=True)
    align.add_argument("--threshold-m", type=float, default=5.0)
    mission = commands.add_parser("reconstruct")
    mission.add_argument("--video", type=Path, required=True)
    mission.add_argument("--telemetry", type=Path, required=True)
    mission.add_argument("--output", type=Path, required=True)
    mission.add_argument("--config", type=Path)
    mission.add_argument("--start-time", help="UTC time of video frame zero")
    mission.add_argument("--full", action="store_true",
                         help="enable dense cloud processing and meshing")
    report = commands.add_parser("report")
    report.add_argument("mission", type=Path)
    depth = commands.add_parser("estimate-depth")
    depth.add_argument("--images", type=Path, required=True)
    depth.add_argument("--output", type=Path, required=True)
    depth.add_argument("--config", type=Path)
    dense = commands.add_parser("reconstruct-dense")
    dense.add_argument("--images", type=Path, required=True)
    dense.add_argument("--sparse-model", type=Path, required=True)
    dense.add_argument("--output", type=Path, required=True)
    dense.add_argument("--config", type=Path)
    cloud = commands.add_parser("process-cloud")
    cloud.add_argument("--input", type=Path, required=True)
    cloud.add_argument("--reference", type=Path, required=True)
    cloud.add_argument("--output", type=Path, required=True)
    cloud.add_argument("--voxel-size-m", type=float, default=0.1)
    mesh = commands.add_parser("mesh")
    mesh.add_argument("--input", type=Path, required=True)
    mesh.add_argument("--output", type=Path, required=True)
    mesh.add_argument("--depth", type=int, default=8)
    mesh.add_argument("--min-points", type=int, default=1000)
    viewer = commands.add_parser("viewer")
    viewer.add_argument("mission", type=Path)
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, default=8765)
    viewer.add_argument("--no-browser", action="store_true")
    masks = commands.add_parser("mask-dynamics")
    masks.add_argument("--images", type=Path, required=True)
    masks.add_argument("--output", type=Path, required=True)
    masks.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "mask-dynamics":
            from .masking import YoloSegmenter, create_masks
            config = load_config(args.config)
            images = sorted(args.images.glob("*.jpg"))
            device = config.segmentation.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            segmenter = YoloSegmenter(config.segmentation.model_id, device,
                                      config.segmentation.confidence)
            artifacts, manifest = create_masks(images, args.output, segmenter,
                                               config.segmentation.batch_size)
            print(json.dumps({"masks": len(artifacts), "manifest": str(manifest)}, indent=2))
            return 0
        if args.command == "viewer":
            from .viewer import serve
            serve(args.mission, args.host, args.port, not args.no_browser)
            return 0
        if args.command == "mesh":
            from .mesh import generate_mesh
            print(json.dumps(generate_mesh(args.input, args.output,
                                           depth=args.depth,
                                           min_points=args.min_points), indent=2))
            return 0
        if args.command == "process-cloud":
            from .pointcloud import process_dense_cloud
            metrics = process_dense_cloud(args.input, args.reference, args.output,
                                          args.voxel_size_m)
            print(json.dumps(metrics, indent=2))
            return 0
        if args.command == "reconstruct-dense":
            config = load_config(args.config)
            root = ensure_output(args.output)
            image_ids = {item.name: file_identifier(item)
                         for item in sorted(args.images.glob("*.jpg"))}
            model_ids = {item.name: file_identifier(item)
                         for item in sorted(args.sparse_model.glob("*.bin"))}
            if not image_ids or not model_ids:
                raise ValueError("Selected frames and a binary COLMAP model are required")
            artifacts = run_stage(
                root, "dense", 1, config.digest_for("colmap"), {**image_ids, **model_ids},
                lambda: [reconstruct_dense(args.images, args.sparse_model,
                                           root / "reconstruction/dense",
                                           discover_colmap(config.colmap.executable))],
            )
            print(json.dumps({"dense_points": ply_vertex_count(artifacts[0]),
                              "coordinate_frame": "COLMAP reconstruction units",
                              "cloud": str(artifacts[0])}, indent=2))
            return 0
        if args.command == "estimate-depth":
            from .depth import DepthAnythingEstimator, cached_depth
            config = load_config(args.config)
            images = sorted(args.images.glob("*.jpg"))
            if not images:
                raise ValueError("No JPEG images found in --images")
            estimator = DepthAnythingEstimator(config.depth.model_id,
                                               config.depth.device,
                                               config.depth.batch_size)
            artifacts = cached_depth(images, args.output, estimator,
                                     config.depth.batch_size)
            print(json.dumps({"relative_depth_maps": len(artifacts),
                              "geometry_class": "inferred",
                              "units": "relative"}, indent=2))
            return 0
        if args.command == "reconstruct":
            config = load_config(args.config)
            start_time = args.start_time or config.gps.video_start_time
            if not start_time:
                raise ValueError("--start-time or gps.video_start_time is required")
            if args.full:
                config.reconstruction.dense = True
                config.reconstruction.process_pointcloud = True
                config.reconstruction.mesh = True
                page = reconstruct_full(args.video, args.telemetry, args.output,
                                        config, start_time)
            elif any((config.reconstruction.dense,
                      config.reconstruction.inferred_depth,
                      config.reconstruction.process_pointcloud,
                      config.reconstruction.mesh)):
                page = reconstruct_full(args.video, args.telemetry, args.output,
                                        config, start_time)
            else:
                page = reconstruct(args.video, args.telemetry, args.output,
                                   config, start_time)
            print(page)
            return 0
        if args.command == "report":
            print(build_report(args.mission))
            return 0
        if args.command == "doctor":
            checks = doctor(args.output)
            for item in checks:
                print(f"{item.status:4} {item.name}: {item.detail}")
            result = doctor_exit(checks)
            print("Overall:", "READY" if result == 0 else "BLOCKED")
            return result
        if args.command == "telemetry":
            samples = load_telemetry(args.path)
            print(json.dumps({"samples": len(samples), "start": samples[0].timestamp,
                              "end": samples[-1].timestamp}, indent=2))
            return 0
        if args.command == "sync-telemetry":
            samples = load_telemetry(args.telemetry)
            output, missing = synchronize(args.manifest, samples,
                                          parse_timestamp(args.start_time), args.output,
                                          args.method, args.tolerance)
            print(json.dumps({"output": str(output), "telemetry_samples": len(samples),
                              "unmatched_frames": missing}, indent=2))
            return 0
        if args.command == "reconstruct-sparse":
            config = load_config(args.config)
            root = ensure_output(args.output)
            images = sorted(args.images.glob("*.jpg"))
            if not images:
                raise ValueError("No JPEG frames found in --images")
            identifiers = {image.name: file_identifier(image) for image in images}
            def action() -> list[Path]:
                reconstruct_sparse(args.images, root / "reconstruction",
                                   discover_colmap(config.colmap.executable),
                                   config.camera.model, config.colmap.use_gpu,
                                   config.colmap.sequential_overlap, None,
                                   config.camera.parameters, config.camera.single_camera)
                model = root / "reconstruction" / "text_model"
                return [model / "images.txt", model / "points3D.txt", model / "cameras.txt"]
            paths = run_stage(root, "sparse", 1, config.digest_for("colmap"), identifiers, action)
            result = read_sparse_text(paths[0].parent)
            print(json.dumps({"registered_images": len(result.poses),
                              "sparse_points": result.point_count,
                              "observations": result.observations,
                              "mean_track_length": result.mean_track_length,
                              "mean_reprojection_error_px":
                                  result.mean_reprojection_error_px}, indent=2))
            return 0
        if args.command == "align-sparse":
            from .geoexport import align_sparse
            root = ensure_output(args.output)
            inputs = {
                "images": file_identifier(args.model / "images.txt"),
                "points": file_identifier(args.model / "points3D.txt"),
                "synchronized": file_identifier(args.synchronized),
            }
            def align_action() -> list[Path]:
                destination = root / "geospatial"
                align_sparse(args.model, args.synchronized, destination, args.threshold_m)
                return [destination / name for name in
                        ("sparse_georeferenced.ply", "camera_poses.json",
                         "trajectory.geojson", "reference.json", "metrics.json")]
            paths = run_stage(root, "align_sparse", 1, str(args.threshold_m), inputs,
                              align_action)
            print(paths[-1].read_text(encoding="utf-8"))
            return 0
        if args.command == "inspect-video":
            print(json.dumps(asdict(inspect_video(args.video)), indent=2))
            return 0
        if args.command == "extract-frames":
            config = load_config(args.config)
            root = ensure_output(args.output)
            paths = run_stage(root, "extract_frames", 1, config.digest_for("video", "frame_selection"),
                              {"video": file_identifier(args.video)},
                              lambda: [extract_frames(args.video, root / "frames", config)])
            print(paths[0])
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
