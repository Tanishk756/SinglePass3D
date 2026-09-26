from pathlib import Path

from desktop_app import find_project_root


def test_desktop_launcher_finds_parent_installation(tmp_path: Path, monkeypatch):
    root = tmp_path / "SinglePass3D"
    (root / ".venv/Scripts").mkdir(parents=True)
    (root / ".venv/Scripts/python.exe").write_bytes(b"")
    (root / "streamlit_app.py").write_text("", encoding="utf-8")
    executable = root / "dist/desktop/SinglePass3D.exe"
    executable.parent.mkdir(parents=True)
    monkeypatch.delenv("SINGLEPASS3D_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_project_root(executable) == root


def test_desktop_launcher_prefers_configured_home(tmp_path: Path, monkeypatch):
    root = tmp_path / "installation"
    (root / ".venv/Scripts").mkdir(parents=True)
    (root / ".venv/Scripts/python.exe").write_bytes(b"")
    (root / "streamlit_app.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("SINGLEPASS3D_HOME", str(root))
    assert find_project_root() == root
