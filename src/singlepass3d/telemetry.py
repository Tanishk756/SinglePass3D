"""Format-neutral drone telemetry loading and UTC synchronization."""
from __future__ import annotations

import csv
import json
from bisect import bisect_left
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .core import validate_input


def parse_timestamp(value: str | float) -> float:
    """Return UTC Unix seconds from ISO-8601 or numeric Unix seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("Telemetry timestamps must include a UTC offset")
        return parsed.astimezone(UTC).timestamp()


class TelemetrySample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: float
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude: float
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    velocity_x: float | None = None
    velocity_y: float | None = None
    velocity_z: float | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def timestamp_to_seconds(cls, value: object) -> float:
        if not isinstance(value, (str, int, float)):
            raise TypeError("timestamp must be ISO-8601 or Unix seconds")
        return parse_timestamp(value)

    @model_validator(mode="after")
    def finite_values(self) -> TelemetrySample:
        import math

        for name, value in self.model_dump().items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


def load_telemetry(path: Path) -> list[TelemetrySample]:
    path = validate_input(path, {".csv", ".json"})
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, list):
            raise ValueError("Telemetry JSON must be an array of samples")
        rows = raw
    if not rows:
        raise ValueError("Telemetry is empty")
    samples = [TelemetrySample.model_validate({
        key: (None if value == "" else value) for key, value in row.items()
    }) for row in rows]
    samples.sort(key=lambda item: item.timestamp)
    if any(left.timestamp == right.timestamp for left, right in pairwise(samples)):
        raise ValueError("Telemetry has duplicate timestamps")
    return samples


def sample_at(samples: list[TelemetrySample], timestamp: float, method: str,
              tolerance: float) -> TelemetrySample | None:
    if not samples:
        return None
    times = [sample.timestamp for sample in samples]
    index = bisect_left(times, timestamp)
    candidates = [samples[i] for i in (index - 1, index) if 0 <= i < len(samples)]
    nearest = min(candidates, key=lambda item: abs(item.timestamp - timestamp))
    if abs(nearest.timestamp - timestamp) > tolerance and (
        method == "nearest" or index == 0 or index == len(samples)
    ):
        return None
    if method == "nearest":
        return nearest
    if method != "interpolate":
        raise ValueError(f"Unknown sync method: {method}")
    if index < len(samples) and times[index] == timestamp:
        return samples[index]
    if index == 0 or index == len(samples):
        return None
    before, after = samples[index - 1], samples[index]
    if timestamp - before.timestamp > tolerance or after.timestamp - timestamp > tolerance:
        return None
    fraction = (timestamp - before.timestamp) / (after.timestamp - before.timestamp)
    values = {}
    for key in TelemetrySample.model_fields:
        if key == "timestamp":
            values[key] = timestamp
        elif key in {"latitude", "longitude", "altitude"}:
            values[key] = getattr(before, key) + fraction * (
                getattr(after, key) - getattr(before, key))
        else:
            first, last = getattr(before, key), getattr(after, key)
            values[key] = None if first is None or last is None else first + fraction * (last - first)
    return TelemetrySample.model_validate(values)


def synchronize(manifest: Path, samples: list[TelemetrySample], start_time: float,
                output: Path, method: str, tolerance: float) -> tuple[Path, int]:
    """Write one row per selected frame; retain unmatched frames with empty GPS."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame_id", "filename", "video_timestamp", "utc_timestamp", "matched",
              "latitude", "longitude", "altitude", "roll", "pitch", "yaw",
              "velocity_x", "velocity_y", "velocity_z"]
    missing = 0
    with manifest.open(newline="", encoding="utf-8-sig") as source, output.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        required = {"frame_id", "filename", "timestamp", "selected"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Frame manifest is missing required columns")
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for frame in reader:
            if frame["selected"].lower() not in {"true", "1"}:
                continue
            video_time = float(frame["timestamp"])
            utc_time = start_time + video_time
            matched = sample_at(samples, utc_time, method, tolerance)
            if matched is None:
                missing += 1
            row = {"frame_id": frame["frame_id"], "filename": frame["filename"],
                   "video_timestamp": video_time, "utc_timestamp": utc_time,
                   "matched": matched is not None}
            if matched is not None:
                row.update({key: getattr(matched, key) for key in fields[5:]})
            writer.writerow(row)
    return output, missing
