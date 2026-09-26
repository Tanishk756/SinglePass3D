param([switch]$NoShortcut)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Run .\setup.ps1 -Full -IncludeCuda first." }
& $python -m pip install "pywebview>=5,<7" "pyinstaller>=6,<7"
if ($LASTEXITCODE -ne 0) { throw "Desktop build dependencies failed to install." }
& $python -m PyInstaller --noconfirm --clean --onefile --windowed --name SinglePass3D --distpath dist\desktop --workpath build\desktop --specpath build desktop_app.py
if ($LASTEXITCODE -ne 0) { throw "Desktop executable build failed." }
$exe = Join-Path $PSScriptRoot "dist\desktop\SinglePass3D.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw "Desktop executable was not created." }
if (-not $NoShortcut) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcut = Join-Path $desktop "SinglePass3D.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($shortcut)
    $link.TargetPath = $exe
    $link.WorkingDirectory = $PSScriptRoot
    $link.Description = "SinglePass3D local video reconstruction studio"
    $link.Save()
    Write-Host "Desktop shortcut created: $shortcut"
}
Write-Host "Desktop executable: $exe"
