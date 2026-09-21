"""Streaming video metadata and selected-frame extraction."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .core import ensure_output, validate_input


@dataclass(frozen=True)
class VideoInfo:
    path: str
    codec: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float

def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: pip install '.[video]'") from exc

def inspect_video(path: Path) -> VideoInfo:
    cv2 = _cv2()
    path = validate_input(path, {".mp4", ".mov"})
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid metadata: {path}")
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * shift)) & 255) for shift in range(4))
        return VideoInfo(str(path), codec, fps, width, height, count, count / fps)
    finally:
        capture.release()

def frame_timestamps(info: VideoInfo, frame_ids: list[int]) -> list[float]:
    return [frame_id / info.fps for frame_id in frame_ids]

def extract_frames(path: Path, output: Path, config: PipelineConfig) -> Path:
    cv2 = _cv2()
    info = inspect_video(path)
    output = ensure_output(output)
    capture = cv2.VideoCapture(info.path)
    manifest = output / "frame_manifest.csv"
    fields = ["frame_id", "filename", "timestamp", "source_timestamp",
              "blur_score", "exposure_score", "selected", "rejection_reason"]
    previous = None
    next_time = 0.0
    try:
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            frame_id = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_id / info.fps
                frame_id += 1
                if timestamp + 1e-9 < next_time:
                    continue
                settings = config.frame_selection
                step = (settings.interval_seconds if settings.strategy == "interval"
                        else 1.0 / settings.target_fps)
                next_time = timestamp + step
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                exposure = float(gray.mean())
                reason = ""
                if blur < settings.blur_threshold:
                    reason = "blur"
                elif exposure < settings.brightness_min or exposure > settings.brightness_max:
                    reason = "exposure"
                elif settings.strategy == "motion" and previous is not None:
                    difference = float(cv2.absdiff(gray, previous).mean())
                    if difference < settings.motion_threshold:
                        reason = "low_motion"
                previous = gray
                filename = f"frame_{frame_id - 1:08d}.jpg" if not reason else ""
                if filename:
                    image = frame
                    maximum = config.video.max_dimension
                    if maximum and max(info.width, info.height) > maximum:
                        scale = maximum / max(info.width, info.height)
                        image = cv2.resize(frame, None, fx=scale, fy=scale)
                    if not cv2.imwrite(str(output / filename), image):
                        raise OSError(f"Failed to write {filename}")
                writer.writerow({"frame_id": frame_id - 1, "filename": filename,
                                 "timestamp": f"{timestamp:.6f}", "source_timestamp": f"{timestamp:.6f}",
                                 "blur_score": f"{blur:.4f}", "exposure_score": f"{exposure:.4f}",
                                 "selected": bool(filename), "rejection_reason": reason})
    finally:
        capture.release()
    return manifest
