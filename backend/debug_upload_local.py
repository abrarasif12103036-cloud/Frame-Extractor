"""Upload a local video to the /upload endpoint with red tracking enabled, save returned zip, and analyze frames for green markers and red pixels.
Usage (run from project root):
  .venv\Scripts\python backend\debug_upload_local.py "C:\path\to\video.mp4"
"""
import sys
import os
import io
import zipfile
import urllib.request
import urllib.parse
from http.client import HTTPResponse

if len(sys.argv) < 2:
    print('Usage: python backend/debug_upload_local.py <video_path>')
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.exists(video_path):
    print('Video not found:', video_path)
    sys.exit(1)

import requests

url = 'http://127.0.0.1:5000/upload'
files = {'video': (os.path.basename(video_path), open(video_path, 'rb'), 'video/mp4')}
data = {'interval': '0.1', 'track_red': '1'}
print('Uploading', video_path)
r = requests.post(url, files=files, data=data, stream=True)
print('Status:', r.status_code)
print('Headers:', r.headers.get('X-Frames-Count'), r.headers.get('X-Markers-Count'), r.headers.get('X-Used-Detector'))
if r.status_code != 200:
    try:
        print('JSON:', r.json())
    except Exception:
        print('Response text (first 500 chars):')
        try:
            print(r.text[:500])
        except Exception:
            print('Response length', len(r.content))
    sys.exit(1)

out_zip = os.path.join(os.getenv('TEMP', '/tmp'), 'user_frames.zip')
with open(out_zip, 'wb') as of:
    for chunk in r.iter_content(chunk_size=8192):
        of.write(chunk)
print('Saved ZIP to', out_zip)

# Analyze zip
import PIL.Image as Image

z = zipfile.ZipFile(out_zip)
allnames = z.namelist()
imgs = [n for n in allnames if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
print('Found images in zip:', len(imgs))
if 'detections.json' in allnames:
    import json
    j = json.loads(z.read('detections.json'))
    print('detections.json sample (first 10):')
    i = 0
    for k, v in list(j.items())[:10]:
        print(' ', k, '->', v)
        i += 1

os.makedirs('backend/debug_outputs', exist_ok=True)

for name in imgs[:12]:
    data = z.read(name)
    img = Image.open(io.BytesIO(data)).convert('RGB')
    w, h = img.size
    pixels = img.load()
    green_coords = []
    red_coords = []
    for y in range(0, h):
        for x in range(0, w):
            rch, gch, bch = pixels[x, y]
            # green detection
            if gch > 150 and gch > rch + 50 and gch > bch + 50:
                green_coords.append((x, y))
            # red detection
            if rch > 120 and rch - max(gch, bch) > 30:
                red_coords.append((x, y))
    print(name, 'size', (w, h), 'green', len(green_coords), 'red', len(red_coords))
    # Save annotated image showing first green and red found
    out = img.copy()
    draw = Image.Image.draw if False else None
    from PIL import ImageDraw
    d = ImageDraw.Draw(out)
    if green_coords:
        gx, gy = green_coords[0]
        d.ellipse((gx-6, gy-6, gx+6, gy+6), outline=(0,255,0), width=2)
    if red_coords:
        rx, ry = red_coords[0]
        d.ellipse((rx-6, ry-6, rx+6, ry+6), outline=(255,0,0), width=2)
    outpath = os.path.join('backend/debug_outputs', name.replace('/', '_'))
    out.save(outpath)
    print('Wrote', outpath)

print('Analysis done. Inspect backend/debug_outputs for annotated frames.')
