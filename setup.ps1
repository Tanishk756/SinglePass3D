param([switch]$IncludeVideo)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = $null
try {
    $python = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
} catch {
    $python = $null
}
if (-not $python) { throw "Python 3.11 is required. Install it from python.org, then rerun setup.ps1." }
& $python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Failed to create Python 3.11 virtual environment." }
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[dev]"
if ($IncludeVideo) { & $venvPython -m pip install -e ".[video]" }
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }
& $venvPython -m singlepass3d.cli doctor
