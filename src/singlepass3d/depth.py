"""Optional relative monocular depth estimation.

Outputs are inferred, relative-depth maps and must never be treated as metric
measurements without independent geometric calibration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .core import ensure_output, file_identifier


@dataclass(frozen=True)
class DepthResult:
    values: np.ndarray
    model: str
    units: str = "relative"
    geometry_class: str = "inferred"

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or not np.isfinite(self.values).all():
            raise ValueError("Depth prediction must be a finite 2D array")
        if self.units != "relative" or self.geometry_class != "inferred":
            raise ValueError("Monocular depth must retain relative and inferred labels")


class DepthEstimator(Protocol):
    model_id: str

    def estimate(self, images: list[Path]) -> list[DepthResult]: ...


class DepthAnythingEstimator:
    """Transformers backend, loaded only when this optional stage is requested."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str = "auto", batch_size: int = 1):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install the ai extra to use Depth Anything") from exc
        selected = "cuda:0" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device)
        if selected.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch CUDA is unavailable")
        self.model_id = model_id
        self.model = pipeline("depth-estimation", model=model_id, device=selected)
        self.batch_size = batch_size

    def estimate(self, images: list[Path]) -> list[DepthResult]:
        results = []
        for offset in range(0, len(images), self.batch_size):
            batch = images[offset:offset + self.batch_size]
            prediction = self.model([str(path) for path in batch],
                                    batch_size=self.batch_size)
            if isinstance(prediction, dict):
                prediction = [prediction]
            for item in prediction:
                depth = item["predicted_depth"]
                values = depth.detach().cpu().numpy().squeeze().astype(np.float32)
                results.append(DepthResult(values=values, model=self.model_id))
        return results


def cached_depth(images: list[Path], output: Path, estimator: DepthEstimator,
                 batch_size: int = 1) -> list[Path]:
    """Cache per-image relative depth without retaining a whole video in memory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = ensure_output(output)
    artifacts: list[Path] = []
    pending: list[tuple[Path, Path]] = []
    for image in images:
        identity = f"{estimator.model_id}|{file_identifier(image)}"
        key = hashlib.sha256(identity.encode()).hexdigest()
        destination = output / f"{key}.npz"
        artifacts.append(destination)
        if not destination.is_file():
            pending.append((image, destination))
    for offset in range(0, len(pending), batch_size):
        group = pending[offset:offset + batch_size]
        predictions = estimator.estimate([item[0] for item in group])
        if len(predictions) != len(group):
            raise ValueError("Depth backend returned an unexpected batch length")
        for (image, destination), result in zip(group, predictions):
            if result.model != estimator.model_id:
                raise ValueError("Depth backend returned the wrong model identifier")
            metadata = json.dumps({"source": str(image.resolve()), "model": result.model,
                                   "units": result.units,
                                   "geometry_class": result.geometry_class})
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, depth=result.values, metadata=metadata)
            temporary.replace(destination)
    return artifacts
