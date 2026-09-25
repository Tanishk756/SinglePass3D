"""Optional dynamic-object masks for reconstruction inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .core import ensure_output

DYNAMIC_CLASSES = frozenset({"person", "bicycle", "car", "motorcycle", "bus", "truck",
                             "bird", "cat", "dog", "horse", "sheep", "cow"})

class Segmenter(Protocol):
    model_id: str
    def masks(self, images: list[Path]) -> list[np.ndarray]: ...

class YoloSegmenter:
    """Ultralytics segmentation backend loaded only when requested."""
    def __init__(self, model_id: str = "yolo11n-seg.pt", device: str = "cpu",
                 confidence: float = 0.35):
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Dynamic masking requires pip install '.[segmentation]'") from exc
        self.model_id, self.device, self.confidence = model_id, device, confidence
        self.model = YOLO(model_id)

    def masks(self, images: list[Path]) -> list[np.ndarray]:
        results = self.model.predict([str(p) for p in images], conf=self.confidence,
                                     device=self.device, verbose=False)
        outputs = []
        for image, result in zip(images, results):
            shape = result.orig_shape
            combined = np.zeros(shape, dtype=np.uint8)
            if result.masks is not None and result.boxes is not None:
                data = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for mask, class_id in zip(data, classes):
                    if result.names[int(class_id)] in DYNAMIC_CLASSES:
                        try:
                            import cv2
                        except ImportError as exc:
                            raise RuntimeError("Mask resizing requires the video extra") from exc
                        resized = cv2.resize(mask, (shape[1], shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                        combined[resized > 0.5] = 255
            outputs.append(combined)
        return outputs

def create_masks(images: list[Path], output: Path, segmenter: Segmenter,
                 batch_size: int = 4) -> tuple[list[Path], Path]:
    """Write COLMAP masks: white usable pixels, black dynamic pixels."""
    if batch_size < 1 or not images:
        raise ValueError("At least one image and a positive batch size are required")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Mask writing requires pip install '.[video]'") from exc
    output = ensure_output(output)
    artifacts, records = [], []
    for offset in range(0, len(images), batch_size):
        group = images[offset:offset + batch_size]
        masks = segmenter.masks(group)
        if len(masks) != len(group):
            raise ValueError("Segmentation backend returned the wrong number of masks")
        for image, dynamic in zip(group, masks):
            if dynamic.ndim != 2:
                raise ValueError("Segmentation masks must be two-dimensional")
            usable = np.where(dynamic > 0, 0, 255).astype(np.uint8)
            destination = output / (image.name + ".png")
            if not cv2.imwrite(str(destination), usable):
                raise OSError(f"Failed to write {destination}")
            artifacts.append(destination)
            records.append({"image": image.name, "mask": destination.name,
                            "dynamic_pixels": int(np.count_nonzero(dynamic)),
                            "total_pixels": int(dynamic.size)})
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"model": segmenter.model_id, "masks": records}, indent=2),
                        encoding="utf-8")
    return artifacts, manifest
