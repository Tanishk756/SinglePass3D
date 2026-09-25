from pathlib import Path

import pytest

from singlepass3d.app_runtime import reconstruction_command, safe_stem, save_upload


def test_safe_upload_and_video_only_command(tmp_path: Path):
    assert safe_stem("../../My Drone (1).MP4") == "my-drone-1"
    video = save_upload(b"video", "../flight.MP4", tmp_path, {".mp4"})
    assert video == tmp_path / "flight.mp4"
    command = reconstruction_command(video, tmp_path / "out", Path("fast.yaml"), True)
    assert "reconstruct-video" in command
    assert command[-1] == "--full"


def test_rejects_unsupported_or_empty_upload(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported"):
        save_upload(b"x", "video.exe", tmp_path, {".mp4"})
    with pytest.raises(ValueError, match="empty"):
        save_upload(b"", "video.mp4", tmp_path, {".mp4"})


def test_metric_command_requires_start_time(tmp_path: Path):
    with pytest.raises(ValueError, match="start time"):
        reconstruction_command(Path("video.mp4"), tmp_path, Path("config.yaml"), False,
                               Path("telemetry.csv"), None)
