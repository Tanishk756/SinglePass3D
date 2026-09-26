"""Safe filesystem and subprocess helpers for the local web application."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

VIDEO_EXTENSIONS = {".mp4", ".mov"}
TELEMETRY_EXTENSIONS = {".csv", ".json"}


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(value).stem).strip("-_")
    return (stem[:60] or "mission").lower()


def save_upload(data: bytes, name: str, destination: Path,
                allowed: set[str]) -> Path:
    extension = Path(name).suffix.lower()
    if extension not in allowed:
        raise ValueError(f"Unsupported file type: {extension or 'none'}")
    if not data:
        raise ValueError("Uploaded file is empty")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (safe_stem(name) + extension)
    path.write_bytes(data)
    return path


def save_upload_stream(stream, name: str, destination: Path,
                       allowed: set[str]) -> Path:
    """Persist an uploaded file without duplicating a large video in memory."""
    extension = Path(name).suffix.lower()
    if extension not in allowed:
        raise ValueError(f"Unsupported file type: {extension or 'none'}")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (safe_stem(name) + extension)
    stream.seek(0)
    with path.open("wb") as target:
        shutil.copyfileobj(stream, target, length=8 * 1024 * 1024)
    if path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty")
    return path


def reconstruction_command(video: Path, output: Path, config: Path, full: bool,
                           telemetry: Path | None = None,
                           start_time: str | None = None) -> list[str]:
    command = [sys.executable, "-m", "singlepass3d.cli"]
    if telemetry is None:
        command.extend(["reconstruct-video", "--video", str(video),
                        "--output", str(output), "--config", str(config)])
    else:
        if not start_time:
            raise ValueError("Video start time is required with telemetry")
        command.extend(["reconstruct", "--video", str(video), "--telemetry",
                        str(telemetry), "--output", str(output), "--config",
                        str(config), "--start-time", start_time])
    if full:
        command.append("--full")
    return command



def _is_process_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (ImportError, psutil.Error, OSError):
        return False


def active_missions(output_root: Path, exclude: Path | None = None) -> list[Path]:
    active = []
    for state_path in output_root.glob("*/app-job.json"):
        mission = state_path.parent
        if exclude is not None and mission.resolve() == exclude.resolve():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _is_process_running(state.get("pid")):
            active.append(mission)
    return active


def _active_detail(mission: Path, checkpoints: list[str]) -> str:
    reconstruction = mission / "reconstruction"
    dense = reconstruction / "dense"
    if (dense / "patch_match_stereo.log").is_file():
        return "Estimating dense multi-view depth on the GPU"
    if (dense / "image_undistorter.log").is_file() and "dense" not in checkpoints:
        return "Preparing images for dense reconstruction"
    if (reconstruction / "mapper.log").is_file() or (
        (reconstruction / "sequential_matcher.log").is_file()
        and not (reconstruction / "text_model/images.txt").is_file()
    ):
        return "Estimating camera poses and optimizing 3D geometry"
    if (reconstruction / "feature_extractor.log").is_file():
        return "Matching visual features between frames"
    if "capture_preflight" in checkpoints:
        return "Extracting GPU visual features"
    if "extract_frames" in checkpoints:
        return "Checking motion, blur, exposure, and parallax"
    if "inspect_video" in checkpoints:
        return "Selecting reconstruction frames"
    return "Inspecting source"

def start_job(command: list[str], mission: Path) -> int:
    competing = active_missions(mission.parent, exclude=mission)
    if competing:
        names = ", ".join(item.name for item in competing[:2])
        raise ValueError(
            f"Another reconstruction is already running ({names}). "
            "Wait for it to finish before starting a GPU-heavy mission."
        )
    mission.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    environment = os.environ.copy()
    logical_cores = os.cpu_count() or 1
    physical_cores = logical_cores
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or logical_cores
    except ImportError:
        pass
    thread_count = str(max(1, physical_cores))
    environment.update({
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OPENCV_LOG_LEVEL": "ERROR",
        "PYTHONUNBUFFERED": "1",
    })
    with (mission / "app-job.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            creationflags=flags, cwd=Path(__file__).parents[2], env=environment,
        )
    state = {"pid": process.pid, "command": command, "started_at": datetime.now(UTC).isoformat()}
    (mission / "app-job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return process.pid


def job_status(mission: Path) -> dict:
    state_path = mission / "app-job.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    pid = state.get("pid")
    running = _is_process_running(pid)
    checkpoints = []
    for path in sorted((mission / "checkpoints").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("complete"):
                checkpoints.append(value.get("stage_name", path.stem))
        except (OSError, json.JSONDecodeError):
            continue
    log_path = mission / "app-job.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")[-12000:] if log_path.is_file() else ""
    metrics_path = mission / "reports/metrics.json"
    accepted = True
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            accepted = metrics.get("quality_gate", {}).get("passed", True)
        except (OSError, json.JSONDecodeError):
            accepted = False
    requested_full = "--full" in state.get("command", [])
    dense_complete = (mission / "checkpoints/dense.json").is_file()
    partial = metrics_path.is_file() and accepted and requested_full and not dense_complete
    complete = metrics_path.is_file() and accepted and not partial
    failed = not running and not complete and not partial and bool(state)
    started = state.get("started_at")
    elapsed_seconds = None
    if started:
        try:
            elapsed_seconds = max(
                0, (datetime.now(UTC) - datetime.fromisoformat(started)).total_seconds()
            )
        except ValueError:
            pass
    competing = active_missions(mission.parent, exclude=mission) if running else []
    return {
        **state, "running": running, "complete": complete, "partial": partial,
        "failed": failed,
        "checkpoints": checkpoints, "log": log,
        "active_detail": _active_detail(mission, checkpoints) if running else None,
        "elapsed_seconds": elapsed_seconds,
        "competing_missions": [item.name for item in competing],
    }


def write_runtime_config(
    base: Path,
    destination: Path,
    camera_model: str,
    camera_parameters: str | None,
    altitude_datum: str,
    geoid_separation_m: float | None,
) -> Path:
    """Create a mission-specific config without mutating shared profiles."""
    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Base profile must be a YAML mapping")
    allowed_models = {
        "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL",
        "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE",
    }
    if camera_model not in allowed_models:
        raise ValueError("Unsupported camera model")
    parameters = camera_parameters.strip() if camera_parameters else None
    if parameters:
        try:
            values = [float(value.strip()) for value in parameters.split(",")]
        except ValueError as exc:
            raise ValueError("Camera parameters must be comma-separated numbers") from exc
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Camera parameters must be finite numbers")
        parameters = ",".join(str(value) for value in values)
    if altitude_datum not in {"ellipsoidal", "orthometric"}:
        raise ValueError("Unsupported altitude datum")
    if altitude_datum == "orthometric" and geoid_separation_m is None:
        raise ValueError("Orthometric altitude requires geoid separation")
    raw.setdefault("camera", {})
    raw["camera"].update({"model": camera_model, "parameters": parameters})
    raw.setdefault("gps", {})
    raw["gps"].update({
        "altitude_datum": altitude_datum,
        "geoid_separation_m": geoid_separation_m,
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return destination
