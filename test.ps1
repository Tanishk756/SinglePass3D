$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Run setup.ps1 first." }
& $python -m pytest
exit $LASTEXITCODE
