import pytest

from singlepass3d.cli import main
from singlepass3d.config import load_config
from singlepass3d.core import (
    Checkpoint,
    ensure_output,
    file_identifier,
    mission_logger,
    run_stage,
    validate_input,
)
from singlepass3d.doctor import Check, doctor_exit
from singlepass3d.video import VideoInfo, frame_timestamps


def test_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("frame_selection:\n  target_fps: 2\n", encoding="utf-8")
    assert load_config(path).frame_selection.target_fps == 2
    assert load_config(path).digest() == load_config(path).digest()
    path.write_text("frame_selection:\n  target_fps: -1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text("surprise: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)

def test_checkpoint_and_runner(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"content")
    inputs = {"video": file_identifier(source)}
    calls = []
    def action():
        calls.append(1)
        artifact = tmp_path / "result.txt"
        artifact.write_text("result", encoding="utf-8")
        return [artifact]
    run_stage(tmp_path, "sample", 1, "abc", inputs, action)
    run_stage(tmp_path, "sample", 1, "abc", inputs, action)
    assert len(calls) == 1
    checkpoint = Checkpoint.model_validate_json(
        (tmp_path / "checkpoints/sample.json").read_text(encoding="utf-8"))
    assert checkpoint.valid_for("sample", 1, "abc", inputs, tmp_path)
    assert not checkpoint.valid_for("sample", 2, "abc", inputs, tmp_path)
    assert not checkpoint.valid_for("sample", 1, "xyz", inputs, tmp_path)
    source.write_bytes(b"changed")
    assert not checkpoint.valid_for("sample", 1, "abc", {"video": file_identifier(source)}, tmp_path)
    run_stage(tmp_path, "sample", 1, "abc", {"video": file_identifier(source)}, action)
    assert len(calls) == 2

def test_paths_and_logging(tmp_path):
    output = ensure_output(tmp_path / "a space")
    assert output.is_dir()
    assert not list(output.glob(".write-test-*"))
    assert mission_logger(output).handlers
    assert (output / "logs/pipeline.log").exists()
    source = tmp_path / "file.mp4"
    source.write_bytes(b"content")
    assert validate_input(source, {".mp4"}) == source.resolve()
    with pytest.raises(ValueError):
        validate_input(source, {".mov"})
    with pytest.raises(FileNotFoundError):
        validate_input(tmp_path / "missing.mp4", {".mp4"})

def test_doctor_classification():
    assert doctor_exit([Check("optional", "WARN", "absent")]) == 0
    assert doctor_exit([Check("required", "FAIL", "absent")]) == 1

def test_cli(tmp_path, capsys):
    result = main(["doctor", "--output", str(tmp_path)])
    assert result in (0, 1)
    assert "Overall:" in capsys.readouterr().out

def test_timestamps():
    info = VideoInfo("video.mp4", "avc1", 25, 1920, 1080, 100, 4)
    assert frame_timestamps(info, [0, 25, 50]) == [0, 1, 2]


def test_stage_specific_config_hash():
    from singlepass3d.config import PipelineConfig
    baseline = PipelineConfig()
    changed = PipelineConfig.model_validate({"depth": {"batch_size": 2}})
    assert baseline.digest() != changed.digest()
    assert baseline.digest_for("video", "frame_selection") == changed.digest_for(
        "video", "frame_selection")
    with pytest.raises(ValueError):
        baseline.digest_for("unknown")


def test_cli_doctor_exit_classes(tmp_path, monkeypatch, capsys):
    from singlepass3d.doctor import Check
    monkeypatch.setattr("singlepass3d.cli.doctor",
                        lambda output: [Check("optional", "WARN", "absent")])
    assert main(["doctor", "--output", str(tmp_path)]) == 0
    assert "READY" in capsys.readouterr().out
    monkeypatch.setattr("singlepass3d.cli.doctor",
                        lambda output: [Check("required", "FAIL", "absent")])
    assert main(["doctor", "--output", str(tmp_path)]) == 1
    assert "BLOCKED" in capsys.readouterr().out


def test_reconstruct_requires_clock_origin(tmp_path, capsys):
    assert main(["reconstruct", "--video", "video.mp4", "--telemetry", "gps.csv",
                 "--output", str(tmp_path)]) == 2
    assert "--start-time" in capsys.readouterr().err
