"""Environment diagnostics with optional-component warnings."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .core import ensure_output


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _find_ffmpeg() -> str | None:
    """Find FFmpeg on PATH or in WinGet's per-user package directory."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(package_root.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"))
    return str(candidates[-1]) if candidates else None

def doctor(output: Path) -> list[Check]:
    checks = []
    version = sys.version_info
    checks.append(Check("Python", "PASS" if version.major == 3 and version.minor == 11 else "FAIL",
                        platform.python_version() + " (Python 3.11 required)"))
    checks.append(Check("OS", "PASS" if os.name == "nt" else "WARN", platform.platform()))
    checks.append(Check("CPU", "PASS", platform.processor() or "processor name unavailable"))
    try:
        import psutil
        checks.append(Check("RAM", "PASS", f"{psutil.virtual_memory().total} bytes"))
    except ImportError:
        checks.append(Check("RAM", "WARN", "Install psutil to report installed RAM"))
    ffmpeg = _find_ffmpeg()
    checks.append(Check("FFmpeg", "PASS" if ffmpeg else "WARN",
                        ffmpeg or "ffmpeg not found"))
    try:
        from .colmap import discover_colmap

        colmap = str(discover_colmap())
    except (OSError, RuntimeError):
        colmap = None
    checks.append(Check("COLMAP", "PASS" if colmap else "WARN",
                        colmap or "COLMAP_EXE/colmap not found"))
    nvidia_smi = shutil.which("nvidia-smi")
    names = []
    if nvidia_smi:
        try:
            query = subprocess.run([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=10, check=False)
            if query.returncode == 0:
                names = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.TimeoutExpired):
            pass
    checks.append(Check("NVIDIA GPU", "PASS" if names else "WARN",
                        ", ".join(names) if names else "No NVIDIA GPU detected"))
    checks.append(Check("CUDA", "WARN", "CUDA runtime detection requires PyTorch"))
    for name, module in [("OpenCV", "cv2"), ("PyTorch", "torch"), ("Open3D", "open3d")]:
        found = importlib.util.find_spec(module) is not None
        checks.append(Check(name, "PASS" if found else "WARN", "installed" if found else "not installed"))
    if importlib.util.find_spec("torch") is not None:
        import torch
        cuda = torch.cuda.is_available()
        checks.append(Check("PyTorch CUDA", "PASS" if cuda else "WARN", str(cuda)))
        checks = [Check("CUDA", "PASS" if cuda else "WARN", str(cuda)) if item.name == "CUDA" else item for item in checks]
    else:
        checks.append(Check("PyTorch CUDA", "WARN", "PyTorch unavailable"))
    try:
        ensure_output(output)
        checks.append(Check("Output", "PASS", str(output.resolve())))
    except (OSError, ValueError) as exc:
        checks.append(Check("Output", "FAIL", str(exc)))
    return checks

def doctor_exit(checks: list[Check]) -> int:
    return 1 if any(item.status == "FAIL" for item in checks) else 0
