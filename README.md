# Camera UI — Frame Extractor

Simple app to upload a video and extract frames every 0.1s (10 FPS).

Quick start (backend):
1. Install ffmpeg and ensure it's on your PATH.
2. Create a Python virtualenv and install requirements:
   python -m venv venv
   venv\Scripts\activate
   pip install -r backend\requirements.txt
3. Run the backend:
   set FLASK_APP=backend.app
   set FLASK_ENV=development
   flask run --port 5000

Open `frontend/index.html` in your browser to upload a video and download the ZIP of frames.

Notes:
- Keep an eye on upload size limits in `backend/app.py` (MAX_CONTENT_LENGTH).
- Check ffmpeg availability with: `curl http://127.0.0.1:5000/info` (returns ffmpeg version if available).
- Run tests: install dev deps and run `pytest backend`.
- ffmpeg is required to extract frames. If you don't have ffmpeg installed, installs:
  - Windows: download from https://ffmpeg.org/download.html or use `choco install ffmpeg` (Chocolatey).
  - macOS: `brew install ffmpeg` (Homebrew).
  - Ubuntu/Debian: `sudo apt install ffmpeg`.
- Quick verification and test video generation (once ffmpeg is installed):
  - Verify: `ffmpeg -version` or `curl http://127.0.0.1:5000/info`
  - Generate 1s test video: `ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=10 sample.mp4`
  - Upload via curl: `curl -F "video=@sample.mp4" http://127.0.0.1:5000/upload -o frames.zip`
- You can run the whole setup and test locally with the helper scripts:
  - `run_local.bat` (double-click or run in PowerShell/CMD)
  - `run_all.ps1` (PowerShell) — creates venv, installs deps, downloads ffmpeg if needed, runs tests, starts server, runs e2e, opens `frontend/index.html`.
- This is a simple demo; for production you'd add authentication, quota limits, and robust cleanup.