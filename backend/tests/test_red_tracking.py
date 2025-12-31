import os
import subprocess
import shutil
import zipfile
import tempfile
import io
from PIL import Image
import requests
import pytest

HOST = 'http://127.0.0.1:5000'


def generate_moving_red_video(path):
    # Create a 1 second video by generating image frames with a moving red square and encoding with ffmpeg
    import tempfile
    from PIL import Image, ImageDraw
    tmpd = tempfile.mkdtemp()
    fps = 10
    frames = fps  # 1 second
    w, h = 160, 120
    for i in range(frames):
        img = Image.new('RGB', (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        # moving square across width
        x = int((w - 8) * (i / max(1, frames - 1)))
        y = 50
        draw.rectangle((x, y, x + 8, y + 8), fill=(255, 0, 0))
        img.save(f'{tmpd}/frame_{i:03d}.png')
    # Use ffmpeg to produce an mp4
    cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', f'{tmpd}/frame_%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', path]
    subprocess.run(cmd, check=True)


def test_track_red_and_mark(tmp_path):
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg not available; skipping e2e')

    video = tmp_path / 'red.mp4'
    out_zip = tmp_path / 'frames.zip'

    generate_moving_red_video(str(video))

    # Ensure server is running; skip test if not
    try:
        resp = requests.get(f'{HOST}/health', timeout=1)
        if resp.status_code != 200:
            pytest.skip('Server not running (health returned {})'.format(resp.status_code))
    except Exception:
        pytest.skip('Server not running; skip integration test')

    with open(video, 'rb') as f:
        r = requests.post(f'{HOST}/upload', files={'video': ('red.mp4', f, 'video/mp4')}, data={'interval': '0.1', 'track_red': '1'}, stream=True)
        # If the server lacks opencv, it will return 500 with error message. Skip in that case.
        if r.status_code == 500:
            try:
                j = r.json()
                if j.get('error') == 'opencv not available':
                    pytest.skip('Server does not have opencv installed; skipping red-tracking e2e test')
            except Exception:
                pass
        r.raise_for_status()
        with open(out_zip, 'wb') as of:
            for chunk in r.iter_content(chunk_size=8192):
                of.write(chunk)

    # Inspect ZIP for frames having green markers (approx detection with PIL)
    found_marker = False
    with zipfile.ZipFile(out_zip, 'r') as z:
        for name in z.namelist():
            if not name.lower().endswith('.jpg'):
                continue
            data = z.read(name)
            img = Image.open(io.BytesIO(data)).convert('RGB')
            pixels = img.getdata()
            for px in pixels:
                rch, gch, bch = px[0], px[1], px[2]
                # detect bright green-ish color (marker we drew)
                if gch > 150 and gch > rch + 50 and gch > bch + 50:
                    found_marker = True
                    break
            if found_marker:
                break

    assert found_marker, 'No green marker detected in frames; red-tracking may have failed'