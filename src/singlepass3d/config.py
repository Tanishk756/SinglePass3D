"""Validated configuration and deterministic hashes."""
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class VideoConfig(StrictModel):
    max_dimension: int | None = Field(default=1920, ge=128)

class FrameConfig(StrictModel):
    strategy: Literal["interval", "target_fps", "motion"] = "target_fps"
    interval_seconds: float = Field(default=1.0, gt=0)
    target_fps: float = Field(default=1.0, gt=0)
    motion_threshold: float = Field(default=12.0, ge=0)
    blur_threshold: float = Field(default=40.0, ge=0)
    brightness_min: float = Field(default=15.0, ge=0, le=255)
    brightness_max: float = Field(default=245.0, ge=0, le=255)

class GPSConfig(StrictModel):
    sync_method: Literal["nearest", "interpolate"] = "interpolate"
    tolerance_seconds: float = Field(default=1.0, gt=0)
    video_start_time: str | None = None

class ColmapConfig(StrictModel):
    executable: str | None = None
    use_gpu: bool = False
    camera_model: str = "SIMPLE_RADIAL"
    sequential_overlap: int = Field(default=10, ge=1)

class DepthConfig(StrictModel):
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: Literal["auto", "cpu", "cuda:0"] = "auto"
    batch_size: int = Field(default=1, ge=1)


class PipelineConfig(StrictModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    frame_selection: FrameConfig = Field(default_factory=FrameConfig)
    gps: GPSConfig = Field(default_factory=GPSConfig)
    colmap: ColmapConfig = Field(default_factory=ColmapConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)

    def digest(self) -> str:
        return self.digest_for(*self.model_fields)

    def digest_for(self, *sections: str) -> str:
        configuration = self.model_dump(mode="json")
        unknown = set(sections) - configuration.keys()
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        selected = {section: configuration[section] for section in sections}
        data = json.dumps(selected, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

def load_config(path: Path | None = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Configuration must be a YAML mapping")
    return PipelineConfig.model_validate(raw)
