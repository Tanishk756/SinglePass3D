# Windows setup

Install Python 3.11 and Git. Run `.\setup.ps1 -IncludeVideo` in PowerShell. The script creates `.venv` and installs base, dev, and optional OpenCV packages. Use `.\run.ps1 doctor` and `.\test.ps1`.

For sparse reconstruction, install a Windows COLMAP build and make `colmap.exe` available through PATH, `COLMAP_EXE`, or YAML configuration. Install `.[geospatial]` for NumPy and pyproj. NVIDIA CUDA is optional; the default COLMAP profile requests CPU features and matching. If choosing CUDA or PyTorch, use versions compatible with the installed NVIDIA driver. Paths with spaces are passed as individual subprocess arguments.

The development environment used for this implementation had Python 3.12 only, so its doctor output is BLOCKED; Python 3.11 execution remains unverified.

If local script execution is restricted, run scripts with a process-local policy: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1`. This does not change the machine's persistent policy.
