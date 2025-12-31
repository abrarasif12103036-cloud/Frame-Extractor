"""Parameter sweep for red-tracking thresholds.
Usage: .venv\Scripts\python backend\parameter_sweep.py "C:\path\to\video.mp4"

This script uploads the video repeatedly with different (min_pixels, red_min) pairs and
collects detection statistics from the returned frames/ detections.json.
"""
import sys
import os
import io
import zipfile
import json
import statistics
import requests

if len(sys.argv) < 2:
    print('Usage: python backend/parameter_sweep.py <video_path> [host]')
    sys.exit(1)

video_path = sys.argv[1]
host = sys.argv[2] if len(sys.argv) >= 3 else 'http://127.0.0.1:5000'

if not os.path.exists(video_path):
    print('Video not found:', video_path)
    sys.exit(1)

# Focused combos by default; allow passing a timeout as the 2nd or 3rd arg
combos = []
# default ranges (tuned narrower for faster runs)
min_pixels_list = [40, 80, 160]
red_min_list = [140, 150, 160]
for mp in min_pixels_list:
    for rm in red_min_list:
        combos.append((mp, rm))

# optional timeout: if third arg is provided and digits, use it; if second arg is digits, treat as timeout
timeout = 120
if len(sys.argv) >= 3:
    if sys.argv[2].isdigit():
        timeout = int(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[3].isdigit():
        timeout = int(sys.argv[3])
print('Using request timeout (s):', timeout)

results = []

for (min_pixels, red_min) in combos:
    print(f'Running min_pixels={min_pixels}, red_min={red_min}...')
    try:
        with open(video_path, 'rb') as f:
            files = {'video': (os.path.basename(video_path), f, 'video/mp4')}
            data = {'interval': '0.1', 'track_red': '1', 'min_pixels': str(min_pixels), 'red_min': str(red_min)}
            r = requests.post(host + '/upload', files=files, data=data, timeout=timeout)
        if r.status_code != 200:
            print('  Server error', r.status_code)
            results.append({'min_pixels': min_pixels, 'red_min': red_min, 'error': r.text[:200]})
            continue
        # open zip
        z = zipfile.ZipFile(io.BytesIO(r.content))
        frames = int(r.headers.get('X-Frames-Count', '0'))
        dets = {}
        if 'detections.json' in z.namelist():
            dets = json.loads(z.read('detections.json').decode('utf-8'))
        detected_count = sum(1 for v in dets.values() if v)
        pixels_list = [v.get('pixels', 0) for v in dets.values() if v]
        cx_list = [v.get('cx') for v in dets.values() if v and 'cx' in v]
        cy_list = [v.get('cy') for v in dets.values() if v and 'cy' in v]
        avg_pixels = statistics.mean(pixels_list) if pixels_list else 0
        std_cx = statistics.pstdev(cx_list) if cx_list else 0
        std_cy = statistics.pstdev(cy_list) if cy_list else 0
        # center fraction: fraction of detections within center box (center +/- 1/3 dims)
        center_box = None
        center_frac = 0
        if cx_list:
            # need image size; pick first image
            imgs = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                info = z.read(imgs[0])
                from PIL import Image
                img = Image.open(io.BytesIO(info))
                w, h = img.size
                cx_img = w / 2.0
                cy_img = h / 2.0
                x0, x1 = cx_img - w/3, cx_img + w/3
                y0, y1 = cy_img - h/3, cy_img + h/3
                in_center = sum(1 for (cx, cy) in zip(cx_list, cy_list) if x0 <= cx <= x1 and y0 <= cy <= y1)
                center_frac = in_center / len(cx_list) if cx_list else 0
        fraction_detected = detected_count / frames if frames else 0
        results.append({
            'min_pixels': min_pixels,
            'red_min': red_min,
            'frames': frames,
            'detected_count': detected_count,
            'fraction_detected': round(fraction_detected, 3),
            'avg_pixels': round(avg_pixels, 1),
            'std_cx': round(std_cx, 1),
            'std_cy': round(std_cy, 1),
            'center_frac': round(center_frac, 3)
        })
    except Exception as e:
        print('  Exception', e)
        results.append({'min_pixels': min_pixels, 'red_min': red_min, 'error': str(e)[:200]})

# sort by fraction_detected desc, center_frac desc, avg_pixels desc
results_sorted = sorted([r for r in results if 'error' not in r], key=lambda x: (x['fraction_detected'], x['center_frac'], x['avg_pixels']), reverse=True)

print('\nSweep results (top 6):')
for r in results_sorted[:6]:
    print(r)

print('\nAll results:')
for r in results:
    print(r)

# Save results
with open('backend/parameter_sweep_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print('\nSaved results to backend/parameter_sweep_results.json')
