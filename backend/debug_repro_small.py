"""Create a small moving-red test video and upload to /upload to reproduce the failing test case.
Usage: .venv\Scripts\python backend\debug_repro_small.py
"""
import os
import subprocess
import tempfile
import requests

tmpd = tempfile.mkdtemp()
fps = 10
frames = fps
w, h = 160, 120
for i in range(frames):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = int((w - 8) * (i / max(1, frames - 1)))
    y = 50
    draw.ellipse((x, y, x + 8, y + 8), fill=(255, 0, 0))
    img.save(os.path.join(tmpd, f'frame_{i:03d}.png'))
video_path = os.path.join(tmpd, 'red.mp4')
import shutil
ffmpeg_exe = shutil.which('ffmpeg')
if not ffmpeg_exe:
    # try the project's tools folder
    candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools', 'ffmpeg-8.0.1-essentials_build', 'bin', 'ffmpeg.exe')
    if os.path.exists(candidate):
        ffmpeg_exe = candidate
if not ffmpeg_exe:
    raise RuntimeError('ffmpeg not found; ensure ffmpeg is on PATH or placed in tools folder')
cmd = [ffmpeg_exe, '-y', '-framerate', str(fps), '-i', os.path.join(tmpd, 'frame_%03d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path]
print('Running ffmpeg ->', ' '.join(cmd))
try:
    subprocess.run(cmd, check=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print('ffmpeg failed:', e.stderr.decode(errors='replace'))
    raise

files = {'video': ('red.mp4', open(video_path, 'rb'), 'video/mp4')}
params = {'interval': '0.1', 'track_red': '1'}
print('Uploading small test video to http://127.0.0.1:5000/upload')
try:
    r = requests.post('http://127.0.0.1:5000/upload', files=files, data=params, timeout=60)
    print('status', r.status_code)
    print('headers:', r.headers.get('X-Frames-Count'), r.headers.get('X-Used-Detector'), r.headers.get('X-Markers-Count'))
    if r.status_code != 200:
        print('Response text (first 2000 chars):')
        print(r.text[:2000])
    else:
        open(os.path.join(tmpd, 'out.zip'), 'wb').write(r.content)
        print('Saved out.zip in', tmpd)
except Exception as e:
    print('Request failed:', e)
    raise
