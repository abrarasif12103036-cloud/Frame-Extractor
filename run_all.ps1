# Run all setup, tests, server, and E2E locally (Windows PowerShell)
# Usage: Right-click -> Run with PowerShell or run: powershell -ExecutionPolicy Bypass -File run_all.ps1

set -e
function Write-Log { param($m) Write-Output "[run_all] $m" }

Push-Location $PSScriptRoot

# 1) Create venv if missing
if (-not (Test-Path ".venv")) {
  Write-Log "Creating virtual environment..."
  python -m venv .venv
} else {
  Write-Log "Virtual environment exists"
}

# 2) Activate venv for this session
Write-Log "Activating virtual environment"
. ".\.venv\Scripts\Activate.ps1"

# 3) Install Python deps
Write-Log "Installing Python dependencies (backend/requirements.txt)..."
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

# 4) Ensure ffmpeg present; if not, download a static build
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Write-Log "ffmpeg not found on PATH. Checking tools folder..."
  $tools = Get-ChildItem -Path .\tools -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $tools) {
    $url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
    $zipOut = "$PWD\ffmpeg.zip"
    Write-Log "Downloading ffmpeg from $url to $zipOut (this may take a moment)"
    Invoke-WebRequest -Uri $url -OutFile $zipOut -UseBasicParsing
    Expand-Archive -Path $zipOut -DestinationPath "$PWD\tools" -Force
    Remove-Item $zipOut -Force
    $tools = Get-ChildItem -Path .\tools -Directory | Select-Object -First 1
  }
  $ffbin = Join-Path $tools.FullName 'bin'
  Write-Log "Adding $ffbin to PATH for this session"
  $env:PATH = $env:PATH + ';' + $ffbin
} else {
  Write-Log "ffmpeg already available"
}

# quick ffmpeg check
Write-Log "ffmpeg version:"; ffmpeg -version | Select-Object -First 1

# 5) Run unit tests
Write-Log "Running unit tests (pytest backend)"
python -m pytest backend -q
if ($LASTEXITCODE -ne 0) { Write-Error "Unit tests failed. Aborting."; exit $LASTEXITCODE }

# 6) Start the Flask server if not already running
$portOpen = (Test-NetConnection -ComputerName 127.0.0.1 -Port 5000).TcpTestSucceeded
if (-not $portOpen) {
  Write-Log "Starting Flask server in background..."
  $p = Start-Process -FilePath "$PWD\.venv\Scripts\python.exe" -ArgumentList "backend/app.py" -WorkingDirectory $PWD -PassThru
  Write-Log "Launched process id: $($p.Id)"
} else {
  Write-Log "Server port 5000 already open"
}

# 7) Wait for server /health
$maxWait = 20
for ($i=0; $i -lt $maxWait; $i++) {
  try {
    $r = Invoke-RestMethod -Uri 'http://127.0.0.1:5000/health' -UseBasicParsing -ErrorAction Stop
    if ($r.status -eq 'ok') { Write-Log 'Server healthy'; break }
  } catch {
    Write-Log "Waiting for server... ($i)"
    Start-Sleep -Seconds 1
  }
}

# 8) Run E2E test
Write-Log "Running E2E test (generate sample video, upload, verify frames)"
python backend\tests\test_e2e.py
if ($LASTEXITCODE -ne 0) { Write-Error "E2E test failed"; exit $LASTEXITCODE }

# 9) Open frontend in default browser
Write-Log "Opening frontend/index.html in default browser"
Start-Process "$PWD\frontend\index.html"

Write-Log "All done. Visit frontend/index.html and upload your own videos or use sample.mp4." 
Pop-Location
