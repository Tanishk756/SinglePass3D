"""Windows-compatible COLMAP command backend and text-model reader."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .core import PipelineError, ensure_output


@dataclass(frozen=True)
class CameraPose:
    image_id: int
    name: str
    quaternion: tuple[float, float, float, float]
    translation: tuple[float, float, float]
    center: tuple[float, float, float]
    observations: int


@dataclass(frozen=True)
class SparseResult:
    poses: list[CameraPose]
    point_count: int
    observations: int
    mean_track_length: float | None
    mean_reprojection_error_px: float | None


def discover_colmap(configured: str | None = None) -> Path:
    candidates = [
        configured,
        os.environ.get("COLMAP_EXE"),
        shutil.which("colmap.exe"),
        shutil.which("colmap"),
        r"C:\Program Files\COLMAP\COLMAP.bat",
        r"C:\Program Files\COLMAP\colmap.exe",
    ]
    for item in candidates:
        if item and Path(item).is_file():
            return Path(item).resolve()
    raise PipelineError("colmap", "COLMAP executable not found",
                        "Install COLMAP for Windows or set COLMAP_EXE")


def run_colmap(executable: Path, arguments: list[str], log: Path) -> None:
    """Invoke COLMAP with an argument array and retain full stage output."""
    ensure_output(log.parent)
    command = [str(executable), *arguments]
    environment = os.environ.copy()
    if executable.name.lower() == "colmap.exe":
        distribution = executable.parent.parent
        plugins = distribution / "plugins"
        environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
        if plugins.is_dir():
            environment["QT_PLUGIN_PATH"] = str(plugins)
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=None, env=environment)
    except OSError as exc:
        raise PipelineError("colmap", str(exc), "Verify the COLMAP executable path") from exc
    log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise PipelineError("colmap", f"{arguments[0]} exited {result.returncode}",
                            f"Inspect {log} and verify image overlap and camera settings")


def _camera_center(q: tuple[float, float, float, float],
                   t: tuple[float, float, float]) -> tuple[float, float, float]:
    """COLMAP world-to-camera pose center = -R(q)^T t."""
    w, x, y, z = q
    matrix = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return tuple(-sum(matrix[row][col] * t[row] for row in range(3))
                 for col in range(3))


def read_sparse_text(model: Path) -> SparseResult:
    images_path = model / "images.txt"
    points_path = model / "points3D.txt"
    if not images_path.is_file() or not points_path.is_file():
        raise PipelineError("colmap", "Sparse text model is incomplete",
                            "Run model_converter after a successful mapper stage")
    image_lines = [line.strip() for line in images_path.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#")]
    if len(image_lines) % 2:
        raise ValueError("COLMAP images.txt has an incomplete image record")
    poses = []
    observations = 0
    for header, points in zip(image_lines[::2], image_lines[1::2]):
        parts = header.split(maxsplit=9)
        if len(parts) != 10:
            raise ValueError("COLMAP images.txt has an invalid pose record")
        q = tuple(float(value) for value in parts[1:5])
        t = tuple(float(value) for value in parts[5:8])
        observed = sum(token != "-1" for token in points.split()[2::3])
        observations += observed
        poses.append(CameraPose(int(parts[0]), parts[9], q, t, _camera_center(q, t),
                                observed))
    point_rows = [line.split() for line in points_path.read_text(
        encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    track_lengths = [(len(row) - 8) // 2 for row in point_rows if len(row) >= 8]
    errors = [float(row[7]) for row in point_rows if len(row) >= 8]
    count = len(point_rows)
    return SparseResult(
        poses, count, observations,
        sum(track_lengths) / count if count else None,
        sum(errors) / count if count else None,
    )


def reconstruct_sparse(images: Path, output: Path, executable: Path,
                       camera_model: str, use_gpu: bool, overlap: int,
                       masks: Path | None = None, camera_params: str | None = None,
                       single_camera: bool = True) -> SparseResult:
    """Feature extraction, sequential matching and incremental mapping."""
    images = images.resolve(strict=True)
    if not images.is_dir() or not any(images.glob("*.jpg")):
        raise PipelineError("colmap", "No selected JPEG frames",
                            "Run extract-frames and inspect frame quality thresholds")
    output = ensure_output(output)
    database = output / "database.db"
    sparse = ensure_output(output / "sparse")
    feature_args = ["feature_extractor", "--database_path", str(database),
                    "--image_path", str(images), "--ImageReader.camera_model",
                    camera_model, "--ImageReader.single_camera", str(int(single_camera)),
                    "--FeatureExtraction.use_gpu", str(int(use_gpu))]
    if camera_params:
        feature_args.extend(["--ImageReader.camera_params", camera_params])
    if masks is not None:
        masks = masks.resolve(strict=True)
        feature_args.extend(["--ImageReader.mask_path", str(masks)])
    run_colmap(executable, feature_args, output / "feature_extractor.log")
    run_colmap(executable, ["sequential_matcher", "--database_path", str(database),
                            "--SequentialMatching.overlap", str(overlap),
                            "--FeatureMatching.use_gpu", str(int(use_gpu))],
               output / "sequential_matcher.log")
    run_colmap(executable, ["mapper", "--database_path", str(database),
                            "--image_path", str(images), "--output_path", str(sparse)],
               output / "mapper.log")
    models = [item for item in sparse.iterdir() if item.is_dir()]
    if not models:
        raise PipelineError("colmap", "Mapper registered no image model",
                            "Check overlap, image quality and camera intrinsics")
    model = max(models, key=lambda item: sum(1 for _ in item.iterdir()))
    text_model = ensure_output(output / "text_model")
    run_colmap(executable, ["model_converter", "--input_path", str(model),
                            "--output_path", str(text_model), "--output_type", "TXT"],
               output / "model_converter.log")
    return read_sparse_text(text_model)
