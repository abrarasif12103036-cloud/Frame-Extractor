# Frame Extractor - Run Guide

## Quick Start (After PC Restart)

Follow these steps to run the Frame Extractor UI locally:

### Step 1: Open PowerShell

Open PowerShell and navigate to the project directory:

```powershell
cd d:\frame-extractor
```

### Step 2: Activate Virtual Environment

Activate the Python virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` prefix in your PowerShell prompt, indicating the virtual environment is active.

### Step 3: Start Flask Backend Server

Run the Flask development server:

```powershell
python backend/app.py
```

You should see output similar to:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**Important:** Keep this PowerShell window open. The server must stay running while you use the application.

### Step 4: Open in Browser

Open your web browser and navigate to:

```
http://localhost:5000
```

The Frame Extractor UI should now load with all features available:
- Video upload
- Frame extraction
- Color tracking
- Coordinate plotting
- Distance conversion
- Data export

---

## First-Time Setup (One-Time Only)

If this is your first time running the application, you may need to install dependencies:

### 1. Create Virtual Environment

```powershell
cd d:\frame-extractor
python -m venv .venv
```

### 2. Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r backend/requirements.txt
```

### 4. Verify FFmpeg Installation

The application requires FFmpeg for video processing. If you haven't installed it, download from:
https://ffmpeg.org/download.html

Make sure FFmpeg is in your system PATH or available in the `tools/` folder.

---

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, you'll see an error. Kill existing processes:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

Then try starting the server again.

### Virtual Environment Issues
If activation fails, try:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Missing Dependencies
If you get import errors, reinstall dependencies:

```powershell
pip install -r backend/requirements.txt --upgrade
```

### Module Not Found Errors
Try clearing pip cache:

```powershell
pip install --no-cache-dir -r backend/requirements.txt
```

---

## Project Structure

```
frame-extractor/
├── backend/
│   ├── app.py              # Flask server
│   ├── requirements.txt    # Python dependencies
│   └── tests/              # Test files
├── frontend/
│   ├── index.html          # Main UI
│   ├── styles.css          # Styling
│   ├── upload.js           # JavaScript logic
└── tools/
    └── ffmpeg-*/           # FFmpeg binaries
```

---

## Features

Once running, you can:

1. **Upload Videos** - Drag and drop or click to select a video file
2. **Extract Frames** - Set interval (default 0.1s = 10 FPS)
3. **Track Objects** - Enable tracking and pick target color
4. **View Data** - See extracted frame coordinates in a table
5. **Plot Coordinates** - Generate X/Y vs Time graphs
6. **Convert Units** - Convert pixel coordinates to millimeters
7. **Export Data** - Download frame data as CSV or frames as ZIP

---

## Development

To modify the frontend:
- Edit `frontend/index.html`, `frontend/styles.css`, or `frontend/upload.js`
- Changes are reflected on refresh (no server restart needed)

To modify the backend:
- Edit files in `backend/`
- Restart the Flask server for changes to take effect (Ctrl+C then rerun)

---

## Support

For issues, check:
1. Flask server is running (Step 3 output)
2. Browser can reach `http://localhost:5000`
3. JavaScript console (F12) for client-side errors
4. PowerShell console for server errors

Made By Md. Abrar Asif
