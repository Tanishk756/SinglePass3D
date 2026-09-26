from pathlib import Path

import pytest

from singlepass3d.app_runtime import (
    job_status,
    reconstruction_command,
    safe_stem,
    save_upload,
    write_runtime_config,
)


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


def test_failed_quality_gate_is_not_complete(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text(
        '{"quality_gate":{"passed":false}}', encoding="utf-8")
    (tmp_path / "app-job.json").write_text('{"pid":999999}', encoding="utf-8")
    status = job_status(tmp_path)
    assert not status["complete"]
    assert status["failed"]


def test_runtime_config_is_mission_specific(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("camera:\n  model: SIMPLE_RADIAL\ngps:\n  sync_method: interpolate\n", encoding="utf-8")
    output = write_runtime_config(
        base, tmp_path / "mission.yaml", "PINHOLE", "1000, 1001, 500, 400",
        "orthometric", 31.2)
    content = output.read_text(encoding="utf-8")
    assert "model: PINHOLE" in content
    assert "1000.0,1001.0,500.0,400.0" in content
    assert "geoid_separation_m: 31.2" in content
    assert "SIMPLE_RADIAL" in base.read_text(encoding="utf-8")


def test_runtime_config_requires_orthometric_separation(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="geoid separation"):
        write_runtime_config(base, tmp_path / "out.yaml", "PINHOLE", None, "orthometric", None)
