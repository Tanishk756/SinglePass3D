"""Native Windows launcher for the local SinglePass3D workstation application."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def find_project_root(executable: Path | None = None) -> Path:
    """Find an installed project without depending on a machine-specific path."""
    configured = os.environ.get("SINGLEPASS3D_HOME")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    if executable is not None:
        candidates.extend([
            executable.parent, executable.parent.parent, executable.parent.parent.parent
        ])
    candidates.extend([Path.cwd(), Path(__file__).resolve().parent])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "streamlit_app.py").is_file() and (
            resolved / ".venv/Scripts/python.exe"
        ).is_file():
            return resolved
    raise FileNotFoundError(
        "SinglePass3D installation was not found. Set SINGLEPASS3D_HOME to the project folder."
    )


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])


def wait_until_ready(url: str, process: subprocess.Popen, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    health = f"{url}/_stcore/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The SinglePass3D engine stopped during startup.")
        try:
            with urllib.request.urlopen(health, timeout=1.0) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("SinglePass3D did not become ready within 45 seconds.")


def stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        capture_output=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def main() -> int:
    import webview

    root = find_project_root(Path(sys.executable) if getattr(sys, "frozen", False) else None)
    python = root / ".venv/Scripts/python.exe"
    port = available_port()
    url = f"http://127.0.0.1:{port}"
    log_dir = root / "data/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "desktop-engine.log"
    command = [
        str(python), "-m", "streamlit", "run", str(root / "streamlit_app.py"),
        "--server.headless=true", f"--server.port={port}",
        "--server.address=127.0.0.1", "--browser.gatherUsageStats=false",
    ]
    environment = os.environ.copy()
    environment["SINGLEPASS3D_DESKTOP"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_until_ready(url, process)
            webview.create_window(
                "SinglePass3D Studio",
                url,
                width=1460,
                height=940,
                min_size=(1080, 700),
                confirm_close=True,
            )
            webview.start(gui="edgechromium", private_mode=True)
        finally:
            stop_process_tree(process)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            0,
            f"{exc}\n\nSee data\\logs\\desktop-engine.log for details.",
            "SinglePass3D could not start",
            0x10,
        )
        raise

