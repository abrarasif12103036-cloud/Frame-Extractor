import os
import shutil
import subprocess
import tempfile
import time
import logging
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys


def find_ffmpeg():
    """Return path to ffmpeg binary if available, prefer system ffmpeg, then bundled or tools copy."""
    candidate = shutil.which('ffmpeg')
    if candidate:
        return candidate
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        p = os.path.join(meipass, 'tools', 'ffmpeg', 'ffmpeg.exe')
        if os.path.exists(p):
            return p
    # fallback to the tools folder next to the repo (useful during development)
    candidate2 = os.path.join(os.path.dirname(__file__), '..', 'tools', 'ffmpeg-8.0.1-essentials_build', 'bin', 'ffmpeg.exe')
    if os.path.exists(candidate2):
        return candidate2
    return None

# Config
import sys

# When packaged by PyInstaller, resources are extracted to _MEIPASS; otherwise use module dir
_base_dir = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(_base_dir, 'uploads')
FRAMES_DIR = os.path.join(_base_dir, 'frames')
# Ensure directories exist (works both packaged and during development)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'mkv', 'webm', 'avi'}
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({'error': 'file too large'}), 413


@app.route('/info')
def info():
    """Return ffmpeg version info if available."""
    ffmpeg_exe = find_ffmpeg()
    if not ffmpeg_exe:
        return jsonify({'ffmpeg': None, 'error': 'ffmpeg not found'}), 500
    try:
        res = subprocess.run([ffmpeg_exe, '-version'], check=True, capture_output=True, text=True)
        return jsonify({'ffmpeg': res.stdout.splitlines()[0]})
    except Exception as e:
        return jsonify({'ffmpeg': None, 'error': str(e)}), 500


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Helper: color parsing and detection helpers
import colorsys

NAMED_COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
}

def parse_color_string(s):
    """Parse a color spec (name, #rgb, #rrggbb, or 'r,g,b') into an (R,G,B) tuple 0-255.
    Returns None on failure.
    """
    if not s:
        return None
    s = str(s).strip().lower()
    if s in NAMED_COLORS:
        return NAMED_COLORS[s]
    if s.startswith('#'):
        hexs = s[1:]
        if len(hexs) == 3:
            hexs = ''.join(ch*2 for ch in hexs)
        if len(hexs) == 6:
            try:
                r = int(hexs[0:2], 16)
                g = int(hexs[2:4], 16)
                b = int(hexs[4:6], 16)
                return (r, g, b)
            except Exception:
                return None
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) == 3:
            try:
                r, g, b = [int(p) for p in parts]
                if all(0 <= v <= 255 for v in (r, g, b)):
                    return (r, g, b)
            except Exception:
                return None
    return None


def rgb_to_hsv_cv(rgb):
    """Convert (R,G,B) 0-255 to OpenCV-style HSV (H:0-179, S,V:0-255)
    Returns (h, s, v) ints.
    """
    r, g, b = rgb
    # colorsys uses 0..1 and H 0..1
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return int(h * 179), int(s * 255), int(v * 255)


def hsv_ranges_for_rgb(rgb, dh=12, ds=80, dv=80):
    """Return a list of (lower, upper) HSV ranges suitable for cv2.inRange
    Handles hue wrap-around for colors near 0 (red).
    Each lower/upper is (H,S,V) with H in 0..179, S/V in 0..255.
    """
    h, s, v = rgb_to_hsv_cv(rgb)
    lower_h = max(0, h - dh)
    upper_h = min(179, h + dh)
    lower = (lower_h, max(30, s - ds), max(30, v - dv))
    upper = (upper_h, min(255, s + ds), min(255, v + dv))
    if lower_h == 0 and h - dh < 0:
        # wrap: produce two ranges
        wrap_low = (179 + (h - dh) + 1, lower[1], lower[2])
        wrap_high = (179, upper[1], upper[2])
        return [(lower, upper), ((0, lower[1], lower[2]), (upper_h, upper[1], upper[2]))]
    # if hue near 179-handled by default single range
    return [(lower, upper)]


def detect_color_pillow_coords(img_p, target_rgb=(255, 0, 0), tol=80, min_pixels=80):
    """Return (cx, cy, pixels_in_component, confidence) or (None, None, count, 0.0).
    Uses connected components on pixels whose Euclidean RGB distance to target <= tol.
    """
    w, h = img_p.size
    pixels = img_p.load()
    matched = []
    best_score = None
    best_xy = None
    t_r, t_g, t_b = target_rgb
    tol2 = tol * tol
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            dr = r - t_r
            dg = g - t_g
            db = b - t_b
            dist2 = dr*dr + dg*dg + db*db
            if dist2 <= tol2:
                matched.append((x, y))
            if best_score is None or dist2 < best_score:
                best_score = dist2
                best_xy = (x, y)
    total_candidates = len(matched)
    if total_candidates < min_pixels:
        # fallback to best single pixel if it's close enough
        if best_xy and best_score is not None and best_score <= (tol2 / 4):
            return float(best_xy[0]), float(best_xy[1]), total_candidates, float(max(0.0, 1.0 - best_score / (tol2 or 1)))
        return None, None, total_candidates, 0.0

    # Connected components (8-neighborhood)
    matched_set = set(matched)
    visited = set()
    components = []
    neighs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for px, py in matched:
        if (px, py) in visited:
            continue
        # BFS/DFS to collect component
        stack = [(px, py)]
        comp = []
        visited.add((px, py))
        while stack:
            cx0, cy0 = stack.pop()
            comp.append((cx0, cy0))
            for dx, dy in neighs:
                nx, ny = cx0 + dx, cy0 + dy
                if (nx, ny) in matched_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        components.append(comp)

    best_comp = max(components, key=len)
    xs = [p[0] for p in best_comp]
    ys = [p[1] for p in best_comp]
    cx = float(sum(xs) / len(xs))
    cy = float(sum(ys) / len(ys))
    confidence = float(len(best_comp)) / float(total_candidates or 1)
    return cx, cy, len(best_comp), confidence

# (old detect_red_pillow_coords kept for compatibility)
def detect_red_pillow_coords(img_p, red_min=150, score_min=15, min_pixels=80):
    return detect_color_pillow_coords(img_p, target_rgb=(255, 0, 0), tol=100, min_pixels=min_pixels)

        stack = [(px, py)]
        comp = []
        visited.add((px, py))
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for dx, dy in neighs:
                nx, ny = x + dx, y + dy
                if (nx, ny) in red_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        components.append(comp)

    if not components:
        return None, None, total_candidates, 0.0

    # Choose largest component
    comp = max(components, key=len)
    # Try circle fit (Kasa method) if numpy is available for higher robustness to rim-only blobs
    try:
        import numpy as _np
        pts = _np.array(comp, dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]
        A = _np.column_stack((2 * x, 2 * y, _np.ones_like(x)))
        b = x * x + y * y
        sol, *_ = _np.linalg.lstsq(A, b, rcond=None)
        cx = float(sol[0])
        cy = float(sol[1])
        confidence = len(comp) / total_candidates if total_candidates else 0.0
        return float(cx), float(cy), len(comp), float(confidence)
    except Exception:
        # fallback: use bounding-box center which is robust to rim-heavy blobs
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        confidence = len(comp) / total_candidates if total_candidates else 0.0
        return float(cx), float(cy), len(comp), float(confidence)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'file type not allowed'}), 400

    # Validate interval parameter (seconds between frames)
    interval_str = request.form.get('interval', '0.1')
    try:
        interval = float(interval_str)
    except ValueError:
        return jsonify({'error': 'invalid interval'}), 400
    if interval < 0.01 or interval > 5.0:
        return jsonify({'error': 'invalid interval', 'detail': 'interval must be between 0.01 and 5.0 seconds'}), 400

    fps = 1.0 / interval
    # Clamp fps to a reasonable maximum to avoid resource exhaustion
    if fps > 120.0:
        fps = 120.0

    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    save_name = f"{timestamp}_{filename}"
    input_path = os.path.join(UPLOAD_DIR, save_name)
    file.save(input_path)

    # Create a temp directory for frames
    with tempfile.TemporaryDirectory(dir=FRAMES_DIR) as temp_frames:
        # Use ffmpeg to extract frames at desired fps (1 / interval seconds)
        out_pattern = os.path.join(temp_frames, 'frame_%05d.jpg')
        fps_filter = f"fps={fps:.6f}"
        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            return jsonify({'error': 'ffmpeg not available on server'}), 500
        cmd = [ffmpeg_exe, '-y', '-i', input_path, '-vf', fps_filter, out_pattern]
        try:
            logger.info('Running ffmpeg with interval=%s (fps=%s): %s', interval, fps, ' '.join(cmd))
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # include ffmpeg stderr for debugging
            return jsonify({'error': 'ffmpeg failed', 'detail': e.stderr}), 500

        # Optionally perform color-dot tracking and mark centers on frames
        # Backwards-compatible support: if track_red is set, default to 'red'.
        track_color = request.form.get('track_color') or ( 'red' if request.form.get('track_red', '0').lower() in ('1','true','yes','on') else None )
        used_detector = 'none'
        markers_count = 0
        detections = {}
        # Parse color (supports named colors, #rrggbb, and 'r,g,b')
        target_rgb = parse_color_string(track_color) if track_color else None
        # allow override of detection params
        min_pixels = int(request.form.get('min_pixels', 80))
        tol = int(request.form.get('color_tol', 80))

        min_pixels_req = int(request.form.get('min_pixels', 40))
        if track_red:
            # Try OpenCV-based detection first (more robust); if not available, use Pillow fallback
            try:
                import cv2
                import numpy as np
                use_cv = True
                logger.info('Using OpenCV for red tracking')
            except Exception:
                use_cv = False
                logger.info('OpenCV not available; using Pillow fallback for red tracking')

            for fname in os.listdir(temp_frames):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                path = os.path.join(temp_frames, fname)
                # default no detection
                detections[fname] = None

                if use_cv:
                    used_detector = 'opencv'
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # red can be at low and high hue boundaries
                    lower1 = np.array([0, 100, 50])
                    upper1 = np.array([10, 255, 255])
                    lower2 = np.array([160, 100, 50])
                    upper2 = np.array([179, 255, 255])
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask1, mask2)
                    # clean up small noise
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(c)
                        # require a minimum area to avoid tiny noisy detections
                        if area >= 50:
                            # use minEnclosingCircle for a robust center and radius (subpixel)
                            (fx, fy), fr = cv2.minEnclosingCircle(c)
                            cx = float(fx)
                            cy = float(fy)
                            r = float(fr)
                            # draw a filled circle with outline for sharper, crisper marker
                            radius_px = max(6, int(min(img.shape[0], img.shape[1]) * 0.012))
                            center_int = (int(round(cx)), int(round(cy)))
                            # filled circle (semi-solid) then thin outline
                            cv2.circle(img, center_int, radius_px + 2, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                            cv2.circle(img, center_int, radius_px + 2, (0, 192, 0), thickness=1, lineType=cv2.LINE_AA)
                            cv2.drawMarker(img, center_int, (0, 128, 0), markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
                            cv2.putText(img, f"({int(round(cx))},{int(round(cy))})", (center_int[0] + 8, center_int[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                            # save with higher JPEG quality when possible
                            try:
                                cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            except Exception:
                                cv2.imwrite(path, img)
                            markers_count += 1
                            detections[fname] = {'cx': cx, 'cy': cy, 'area': area, 'radius': r}
                else:
                    used_detector = 'pillow'
                    # Pillow fallback: scan pixels for red-ish colors and compute centroid using clustering
                    try:
                        from PIL import Image, ImageDraw
                    except Exception:
                        continue
                    img_p = Image.open(path).convert('RGB')
                    w, h = img_p.size
                    pixels = img_p.load()
                    red_pixels = []
                    best_score = 0
                    best_xy = None
                    # thresholds and min pixels (can be tuned via request form)
                    red_min = int(request.form.get('red_min', 150))
                    score_min = int(request.form.get('score_min', 25))
                    min_pixels = int(request.form.get('min_pixels', 80))
                    confidence = 0.0
                    for y in range(h):
                        for x in range(w):
                            rch, gch, bch = pixels[x, y]
                            score = rch - max(gch, bch)
                            if rch > red_min and score > 10:
                                red_pixels.append((x, y, score))
                            if score > best_score:
                                best_score = score
                                best_xy = (x, y, score)
                    cx = cy = None
                    if len(red_pixels) >= min_pixels:
                        # spatial bucketing with score aggregation to find the most likely blob
                        bucket_size = max(6, min(w, h) // 150)
                        buckets = {}
                        for (x, y, sc) in red_pixels:
                            key = (x // bucket_size, y // bucket_size)
                            entry = buckets.setdefault(key, {'count': 0, 'score': 0, 'pixels': []})
                            entry['count'] += 1
                            entry['score'] += sc
                            if len(entry['pixels']) < 500:
                                entry['pixels'].append((x, y))
                        # evaluate buckets: prefer those with high average score, sufficient size, and compactness
                        candidates = []
                        for k, v in buckets.items():
                            avg_score = v['score'] / v['count']
                            pts = v['pixels']
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) if pts else 1
                            density = v['count'] / bbox_area if bbox_area > 0 else 0
                            # metric balances strength, size and compactness
                            metric = avg_score * density * v['count']
                            candidates.append((k, v['count'], avg_score, density, metric, pts))
                        # bias metric toward candidates near the image center (most games keep the subject near center)
                        cx_img = w / 2.0
                        cy_img = h / 2.0
                        weighted = []
                        for c in candidates:
                            key, cnt, avg, dens, metric, pts = c
                            # distance to center
                            px = int(sum(p[0] for p in pts) / len(pts))
                            py = int(sum(p[1] for p in pts) / len(pts))
                            d = ((px - cx_img) ** 2 + (py - cy_img) ** 2) ** 0.5
                            center_weight = max(0.2, 1.0 - (d / max(w, h)))
                            weighted_metric = metric * center_weight
                            weighted.append((weighted_metric, cnt, avg, dens, pts, px, py))
                        weighted.sort(key=lambda t: t[0], reverse=True)
                        if weighted:
                            best = weighted[0]
                            best_cnt, best_avg, best_density = best[1], best[2], best[3]
                            pts = best[4]
                            if best_cnt >= min_pixels and best_avg >= 14 and best_density > 0.01:
                                cx = int(sum(p[0] for p in pts) / len(pts))
                                cy = int(sum(p[1] for p in pts) / len(pts))
                            elif best_cnt >= max(10, min_pixels // 2) and best_avg >= 16 and best_density > 0.005:
                                cx = int(sum(p[0] for p in pts) / len(pts))
                                cy = int(sum(p[1] for p in pts) / len(pts))
                    elif best_xy and best_score > (score_min + 15):
                        # very strong single-pixel signal fallback (best_xy stores (x,y,score))
                        cx, cy = best_xy[0], best_xy[1]

                    # Try connected-component detector and prefer its centroid if available
                    try:
                        d_cx, d_cy, d_count, d_conf = detect_red_pillow_coords(img_p, red_min=red_min, score_min=score_min, min_pixels=min_pixels)
                        if d_cx is not None:
                            cx, cy = d_cx, d_cy
                            confidence = d_conf
                    except Exception:
                        pass

                    if cx is not None and cy is not None:
                        # draw a filled, semi-opaque circle plus a crisp outline using RGBA overlay
                        radius_px = max(6, int(min(w, h) * 0.012))
                        img_p_rgba = img_p.convert('RGBA')
                        overlay = Image.new('RGBA', img_p_rgba.size, (0, 0, 0, 0))
                        draw = ImageDraw.Draw(overlay)
                        cx_f = float(cx)
                        cy_f = float(cy)
                        cx_i = int(round(cx_f))
                        cy_i = int(round(cy_f))
                        draw.ellipse((cx_i - radius_px, cy_i - radius_px, cx_i + radius_px, cy_i + radius_px), fill=(0, 255, 0, 140), outline=(0, 180, 0))
                        draw.line((cx_i - radius_px, cy_i, cx_i + radius_px, cy_i), fill=(0, 180, 0), width=1)
                        draw.line((cx_i, cy_i - radius_px, cx_i, cy_i + radius_px), fill=(0, 180, 0), width=1)
                        draw.text((cx_i + 8, cy_i - 12), f"({cx_i},{cy_i})", fill=(0, 255, 0))
                        out = Image.alpha_composite(img_p_rgba, overlay).convert('RGB')
                        # save with higher JPEG quality to reduce compression blur
                        out.save(path, quality=95)
                        markers_count += 1
                        detections[fname] = {'cx': float(cx_f), 'cy': float(cy_f), 'pixels': len(red_pixels), 'confidence': float(confidence)}

            # Write per-frame detections (if any) so clients can inspect results
        try:
            import json
            det_path = os.path.join(temp_frames, 'detections.json')
            with open(det_path, 'w', encoding='utf-8') as df:
                json.dump(detections, df, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Zip the frames
        zip_base = os.path.join(UPLOAD_DIR, f"frames_{timestamp}")
        zip_path = shutil.make_archive(zip_base, 'zip', temp_frames)

        # Count frames that were actually produced in the temp_frames dir and log details
        frames_count = sum(1 for f in os.listdir(temp_frames) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
        logger.info('Extracted %s frames: %s', frames_count, ','.join(sorted([f for f in os.listdir(temp_frames) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])))

    # Optionally remove the uploaded video to save space
    try:
        os.remove(input_path)
    except Exception:
        pass

    # Add headers to indicate detection metadata
    headers = {
        'X-Frames-Count': str(frames_count),
        'X-Markers-Count': str(markers_count),
        'X-Used-Detector': used_detector
    }
    resp = send_file(zip_path, as_attachment=True)
    resp.headers.update(headers)
    return resp


@app.route('/demo', methods=['POST', 'GET'])
def demo():
    # Create a short test video (1s) with a moving colored square (color from query), run processing, and return frames.zip
    color_param = request.args.get('color') or request.args.get('track_color') or 'red'
    target_rgb = parse_color_string(color_param) or (255, 0, 0)
    with tempfile.TemporaryDirectory(dir=FRAMES_DIR) as tmpd:
        video_path = os.path.join(tmpd, 'demo.mp4')
        # Generate frames and encode with ffmpeg
        w, h = 160, 120
        fps = 10
        frames = fps
        png_dir = os.path.join(tmpd, 'pngs')
        os.makedirs(png_dir, exist_ok=True)
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return jsonify({'error': 'Pillow not available'}), 500
        for i in range(frames):
            img = Image.new('RGB', (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            x = int((w - 8) * (i / max(1, frames - 1)))
            y = 50
            draw.rectangle((x, y, x + 8, y + 8), fill=target_rgb)
            img.save(os.path.join(png_dir, f'frame_{i:03d}.png'))
        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            return jsonify({'error': 'ffmpeg not available', 'detail': 'ffmpeg binary not found'}), 500
        cmd = [ffmpeg_exe, '-y', '-framerate', str(fps), '-i', os.path.join(png_dir, 'frame_%03d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return jsonify({'error': 'ffmpeg failed', 'detail': e.stderr.decode(errors='replace')}), 500

        # Now extract frames and run tracking (reuse logic)
        out_pattern = os.path.join(tmpd, 'frame_%05d.jpg')
        try:
            subprocess.run([ffmpeg_exe, '-y', '-i', video_path, '-vf', 'fps=10', out_pattern], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return jsonify({'error': 'ffmpeg failed', 'detail': e.stderr.decode(errors='replace')}), 500

        # run tracking like above
        used_detector = 'none'
        markers_count = 0
        try:
            import cv2
            import numpy as np
            use_cv = True
        except Exception:
            use_cv = False
        for fname in os.listdir(tmpd):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(tmpd, fname)
            if use_cv and target_rgb is not None:
                used_detector = 'opencv'
                img = cv2.imread(path)
                if img is None:
                    continue
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                ranges = hsv_ranges_for_rgb(target_rgb)
                mask = None
                for (low, high) in ranges:
                    low_arr = np.array(low, dtype=np.uint8)
                    high_arr = np.array(high, dtype=np.uint8)
                    m = cv2.inRange(hsv, low_arr, high_arr)
                    if mask is None:
                        mask = m
                    else:
                        mask = cv2.bitwise_or(mask, m)
                if mask is None:
                    continue
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M.get('m00', 0):
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(img, (cx, cy), 6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                        cv2.drawMarker(img, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=1)
                        cv2.putText(img, f"({cx},{cy})", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.imwrite(path, img)
                        markers_count += 1
            elif target_rgb is not None:
                used_detector = 'pillow'
                from PIL import Image, ImageDraw
                img_p = Image.open(path).convert('RGB')
                cx, cy, count, conf = detect_color_pillow_coords(img_p, target_rgb=target_rgb, tol=tol, min_pixels=min_pixels)
                if cx is not None and cy is not None:
                    draw = ImageDraw.Draw(img_p)
                    radius = 6
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(0, 255, 0), width=2)
                    draw.line((cx - 8, cy, cx + 8, cy), fill=(0, 255, 0), width=1)
                    draw.line((cx, cy - 8, cx, cy + 8), fill=(0, 255, 0), width=1)
                    draw.text((cx + 8, cy - 12), f"({int(cx)},{int(cy)})", fill=(0, 255, 0))
                    img_p.save(path)
                    markers_count += 1
            else:
                used_detector = 'pillow'
                from PIL import Image, ImageDraw
                img_p = Image.open(path).convert('RGB')
                w, h = img_p.size
                pixels = img_p.load()
                xs = []
                ys = []
                best_score = 0
                best_xy = None
                for y in range(h):
                    for x in range(w):
                        r, g, b = pixels[x, y]
                        score = r - max(g, b)
                        if r > 100 and score > 10:
                            xs.append(x)
                            ys.append(y)
                        if score > best_score:
                            best_score = score
                            best_xy = (x, y)
                if xs and ys:
                    cx = int(sum(xs) / len(xs))
                    cy = int(sum(ys) / len(ys))
                elif best_xy and best_score > 15:
                    cx, cy = best_xy
                else:
                    cx = cy = None
                if cx is not None and cy is not None:
                    draw = ImageDraw.Draw(img_p)
                    radius = 6
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(0, 255, 0), width=2)
                    draw.line((cx - 8, cy, cx + 8, cy), fill=(0, 255, 0), width=1)
                    draw.line((cx, cy - 8, cx, cy + 8), fill=(0, 255, 0), width=1)
                    draw.text((cx + 8, cy - 12), f"({cx},{cy})", fill=(0, 255, 0))
                    img_p.save(path)
                    markers_count += 1

        zip_base = os.path.join(UPLOAD_DIR, f"demo_{int(time.time())}")
        # Create a frames-only directory so the zip contains only the extracted frames (no source pngs folder)
        frames_only = os.path.join(tmpd, 'frames_only')
        os.makedirs(frames_only, exist_ok=True)
        for f in os.listdir(tmpd):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(tmpd, f), os.path.join(frames_only, f))
        zip_path = shutil.make_archive(zip_base, 'zip', frames_only)
        frames_count = sum(1 for f in os.listdir(frames_only) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        logger.info('Demo produced %s frames in %s (archived from %s)', frames_count, tmpd, frames_only)
        headers = {
            'X-Frames-Count': str(frames_count),
            'X-Markers-Count': str(markers_count),
            'X-Used-Detector': used_detector,
            'X-Track-Color': (track_color or '')
        }
        resp = send_file(zip_path, as_attachment=True)
        resp.headers.update(headers)
        return resp


if __name__ == '__main__':
    app.run(debug=True)
