"""Safe filesystem and subprocess helpers for the local web application."""
from __future__ import annotations

import json
import math
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


def start_job(command: list[str], mission: Path) -> int:
    mission.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with (mission / "app-job.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=flags, cwd=Path(__file__).parents[2])
    state = {"pid": process.pid, "command": command, "started_at": datetime.now(UTC).isoformat()}
    (mission / "app-job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return process.pid


def job_status(mission: Path) -> dict:
    state_path = mission / "app-job.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    pid = state.get("pid")
    running = False
    if pid:
        try:
            import psutil
        except ImportError:
            psutil = None
        if psutil is not None:
            try:
                process = psutil.Process(pid)
                running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except (psutil.Error, OSError):
                running = False
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
    complete = metrics_path.is_file() and accepted
    failed = not running and not complete and bool(state)
    return {**state, "running": running, "complete": complete, "failed": failed,
            "checkpoints": checkpoints, "log": log}


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
