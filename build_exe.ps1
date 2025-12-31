# Build a single .exe using PyInstaller
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\build_exe.ps1

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-Not (Test-Path $venvPy)) {
  Write-Host ".venv not found. Make sure virtualenv exists and dependencies are installed."; exit 1
}

# Ensure PyInstaller is installed
& $venvPy -m pip install pyinstaller --upgrade

# Prepare --add-data for frontend and any other static resources
$addData = "--add-data `"$PSScriptRoot\frontend;frontend`""

# If you have an ffmpeg binary in tools, include it
$ffmpegCandidate = Join-Path $PSScriptRoot "tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
$addBinary = ""
if (Test-Path $ffmpegCandidate) {
  $addBinary = "--add-binary `"$ffmpegCandidate;tools\ffmpeg\`""
  Write-Host "Including ffmpeg binary in bundle"
} else {
  Write-Host "ffmpeg not found in tools, the exe will expect ffmpeg on PATH or in tools folder at runtime"
}

# Build command
$spec = "--onefile $addData $addBinary launcher.py"
Write-Host "Running PyInstaller..."
& $venvPy -m pyinstaller --clean --noconfirm --log-level=INFO $addData $addBinary --name frame-extractor launcher.py

if ($LASTEXITCODE -ne 0) { Write-Host "PyInstaller failed with exit code $LASTEXITCODE"; exit $LASTEXITCODE }

Write-Host "Build complete. Bundled exe: dist\frame-extractor.exe"
Write-Host "Tip: If ffmpeg is not bundled, place ffmpeg.exe next to the exe or add it to PATH."