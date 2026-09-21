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
    for name, command in [("FFmpeg", "ffmpeg"), ("COLMAP", "colmap")]:
        found = shutil.which(command)
        checks.append(Check(name, "PASS" if found else "WARN", found or f"{command} not found"))
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
