import csv

import numpy as np
import pytest

from singlepass3d.config import PipelineConfig
from singlepass3d.video import extract_frames, inspect_video

cv2 = pytest.importorskip("cv2")

def test_real_mp4_inspection_and_streaming_extraction(tmp_path):
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot encode MP4")
    rng = np.random.default_rng(7)
    for index in range(20):
        frame = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        cv2.putText(frame, str(index), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255, 255, 255), 1)
        writer.write(frame)
    writer.release()
    info = inspect_video(video)
    assert info.width == 64 and info.height == 48
    assert info.frame_count == 20
    config = PipelineConfig.model_validate({
        "frame_selection": {"strategy": "target_fps", "target_fps": 2,
                            "blur_threshold": 0, "brightness_min": 0,
                            "brightness_max": 255}})
    manifest = extract_frames(video, tmp_path / "frames", config)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    selected = [row for row in rows if row["selected"] == "True"]
    assert 3 <= len(selected) <= 5
    assert all((manifest.parent / row["filename"]).is_file() for row in selected)


def test_adaptive_selection_limits_near_duplicates(tmp_path):
    video = tmp_path / "adaptive.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot encode MP4")
    rng = np.random.default_rng(11)
    base = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    for index in range(20):
        frame = np.roll(base, index // 4, axis=1)
        writer.write(frame)
    writer.release()
    config = PipelineConfig.model_validate({
        "frame_selection": {
            "strategy": "adaptive", "target_fps": 10,
            "adaptive_min_interval_seconds": 0.3,
            "adaptive_max_interval_seconds": 1.0,
            "motion_threshold": 2, "blur_threshold": 0,
            "brightness_min": 0, "brightness_max": 255,
        }
    })
    manifest = extract_frames(video, tmp_path / "frames", config)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    selected = [row for row in rows if row["selected"] == "True"]
    assert len(selected) < 20
    assert any(row["rejection_reason"] == "near_duplicate" for row in rows)
