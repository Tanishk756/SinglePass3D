"""Checkpoint contract, stage runner and mission logging."""
from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class PipelineError(RuntimeError):
    def __init__(self, stage: str, cause: str, recovery: str):
        super().__init__(f"{stage}: {cause}. Recovery: {recovery}")
        self.stage = stage
        self.recovery = recovery

class Checkpoint(BaseModel):
    stage_name: str
    stage_version: int
    schema_version: int = 1
    configuration_hash: str
    input_identifiers: dict[str, str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    artifact_paths: list[str]
    complete: bool
    duration_seconds: float | None = None

    def valid_for(self, stage: str, version: int, config_hash: str,
                  inputs: dict[str, str], root: Path) -> bool:
        return (self.complete and self.stage_name == stage and self.stage_version == version
                and self.schema_version == 1 and self.configuration_hash == config_hash
                and self.input_identifiers == inputs
                and all((root / item).is_file() for item in self.artifact_paths))

def file_identifier(path: Path) -> str:
    path = path.resolve(strict=True)
    stat = path.stat()
    fingerprint = hashlib.sha256()
    with path.open("rb") as stream:
        fingerprint.update(stream.read(65536))
        if stat.st_size > 65536:
            stream.seek(max(0, stat.st_size - 65536))
            fingerprint.update(stream.read(65536))
    return f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{fingerprint.hexdigest()}"

def validate_input(path: Path, extensions: set[str]) -> Path:
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in extensions:
        raise ValueError(f"Expected file with extension {sorted(extensions)}: {path}")
    return path

def ensure_output(path: Path) -> Path:
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".write-test-{os.getpid()}"
    try:
        with probe.open("x", encoding="utf-8") as stream:
            stream.write("ok")
    finally:
        probe.unlink(missing_ok=True)
    return path

def mission_logger(root: Path) -> logging.Logger:
    logs = ensure_output(root / "logs")
    name = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    logger = logging.getLogger(f"singlepass3d.{name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logs / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    return logger

def run_stage(root: Path, stage: str, version: int, config_hash: str,
              inputs: dict[str, str], action: Callable[[], list[Path]]) -> list[Path]:
    checkpoint_dir = ensure_output(root / "checkpoints")
    checkpoint_path = checkpoint_dir / f"{stage}.json"
    if checkpoint_path.is_file():
        prior = Checkpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
        if prior.valid_for(stage, version, config_hash, inputs, root):
            return [root / item for item in prior.artifact_paths]
    logger = mission_logger(root)
    started = time.perf_counter()
    try:
        artifacts = action()
        relative = [str(path.resolve().relative_to(root.resolve())) for path in artifacts]
        if not all(path.is_file() for path in artifacts):
            raise ValueError("Stage did not produce every declared artifact")
        checkpoint = Checkpoint(stage_name=stage, stage_version=version,
                                configuration_hash=config_hash, input_identifiers=inputs,
                                artifact_paths=relative, complete=True,
                                duration_seconds=time.perf_counter() - started)
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(checkpoint_path)
        logger.info("stage=%s complete artifacts=%s", stage, relative)
        return artifacts
    except Exception:
        logger.exception("stage=%s failed; inspect inputs and output", stage)
        raise
