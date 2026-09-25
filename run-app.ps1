$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Environment not found. Run .\setup.ps1 -Full first."
}
& $python -m streamlit run streamlit_app.py
exit $LASTEXITCODE
