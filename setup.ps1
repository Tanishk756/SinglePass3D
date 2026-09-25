param(
    [switch]$IncludeVideo,
    [switch]$IncludeGeospatial,
    [switch]$IncludeAI,
    [switch]$IncludePointCloud,
    [switch]$IncludeSegmentation,
    [switch]$IncludeCuda,
    [switch]$Full
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = $null
try { $python = & py -3.11 -c "import sys; print(sys.executable)" 2>$null } catch { $python = $null }
if (-not $python) { throw "Python 3.11 is required. Install it from python.org, then rerun setup.ps1." }
& $python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python 3.11 environment." }
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
if ($IncludeCuda) {
    & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
}
$extras = [System.Collections.Generic.List[string]]::new()
$extras.Add("dev")
$extras.Add("viewer")
if ($IncludeVideo -or $Full) { $extras.Add("video") }
if ($IncludeGeospatial -or $Full) { $extras.Add("geospatial") }
if ($IncludeAI -or $Full) { $extras.Add("ai") }
if ($IncludePointCloud -or $Full) { $extras.Add("pointcloud"); $extras.Add("mesh") }
if ($IncludeSegmentation -or $Full) { $extras.Add("segmentation") }
$extraList = ($extras | Select-Object -Unique) -join ","
$installTarget = ".[" + $extraList + "]"
& $venvPython -m pip install -e $installTarget
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }
& $venvPython -m singlepass3d.cli doctor
exit $LASTEXITCODE
