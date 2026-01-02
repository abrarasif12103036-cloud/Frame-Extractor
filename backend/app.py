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

# Serve frontend from parent directory
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app = Flask(__name__, static_folder=frontend_path, static_url_path='')
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
    """Parse a color spec (name, #rgb, #rrggbb, rgb, rrggbb, or 'r,g,b') into an (R,G,B) tuple 0-255.
    Returns None on failure.
    """
    if not s:
        return None
    s = str(s).strip().lower()
    if s in NAMED_COLORS:
        return NAMED_COLORS[s]
    
    # Handle hex with or without # prefix
    hexs = s.lstrip('#')
    if len(hexs) in (3, 6) and all(c in '0123456789abcdef' for c in hexs):
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


def detect_color_simple_effective(img, target_rgb, color_tolerance=20, min_pixels=80):
    """SIMPLE BUT EFFECTIVE color detection using direct RGB matching with morphology.
    Fast and reliable for most use cases.
    Uses STRICT color matching for eyedropper-sampled colors.
    Returns (cx, cy, pixel_count, confidence) or (None, None, 0, 0.0).
    """
    import cv2
    import numpy as np
    
    h, w = img.shape[:2]
    
    # Extract target color in BGR
    target_b, target_g, target_r = target_rgb[2], target_rgb[1], target_rgb[0]
    
    # Create mask: pixels within tolerance of target color
    # Calculate difference for each channel
    img_f = img.astype(np.float32)
    target_f = np.array([[[target_b, target_g, target_r]]], dtype=np.float32)
    
    # Euclidean distance in BGR space
    diff = img_f - target_f
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # STRICT threshold mapping - much stricter than before!
    # These values are for eyedropper-picked colors from actual video
    # tolerance 10 -> threshold ~20 (very strict)
    # tolerance 20 -> threshold ~35 (strict - default)
    # tolerance 30 -> threshold ~50 (moderate-strict)
    if color_tolerance <= 10:
        threshold = 20
    elif color_tolerance <= 15:
        threshold = 25
    elif color_tolerance <= 20:
        threshold = 35
    elif color_tolerance <= 25:
        threshold = 45
    elif color_tolerance <= 30:
        threshold = 55
    else:
        threshold = 70
    
    # Create binary mask
    mask = (distance <= threshold).astype(np.uint8) * 255
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.info(f'Simple detection: No contours found (tolerance={color_tolerance}, threshold={threshold})')
        return None, None, 0, 0.0
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < min_pixels:
        logger.info(f'Simple detection: Area {area} < min_pixels {min_pixels}')
        return None, None, 0, 0.0
    
    # Calculate centroid
    M = cv2.moments(largest_contour)
    if M['m00'] <= 0:
        return None, None, 0, 0.0
    
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    
    # Confidence: based on area
    max_area = h * w
    confidence = min(0.95, max(0.5, area / (max_area * 0.1)))
    
    # Flip Y-axis: Y extends upward from bottom
    cy = h - cy
    
    logger.info(f'Simple detection: centroid=({cx:.1f},{cy:.1f}), area={area}, threshold={threshold}, confidence={confidence:.3f}')
    
    return float(cx), float(cy), int(area), float(confidence)


def detect_color_ultimate_accuracy(img, target_rgb, color_tolerance=20, min_pixels=80):
    """ULTIMATE ACCURACY color detection using hybrid multi-space approach.
    Combines Lab, HSV, RGB spaces + spatial coherence + adaptive tolerance.
    Returns (cx, cy, pixel_count, confidence) or (None, None, 0, 0.0).
    """
    import cv2
    import numpy as np
    
    h, w = img.shape[:2]
    
    # STAGE 1: Multi-space color matching
    # Lab space (perceptually uniform)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    target_bgr = np.uint8([[[target_rgb[2], target_rgb[1], target_rgb[0]]]])
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0, 0]
    target_l = int(target_lab[0])
    target_a = int(target_lab[1])
    target_b = int(target_lab[2])
    
    # HSV space (hue-based)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    target_h = int(target_hsv[0])
    target_s = int(target_hsv[1])
    target_v = int(target_hsv[2])
    
    # Compute Lab delta ranges
    if color_tolerance <= 10:
        dl, da, db = 8, 10, 10
    elif color_tolerance <= 20:
        dl, da, db = 10, 12, 12
    elif color_tolerance <= 30:
        dl, da, db = 12, 15, 15
    else:
        dl, da, db = 15, 18, 18
    
    # Lab mask: strict color matching
    lab_lower = np.array([max(0, target_l - dl), max(0, target_a - da), max(0, target_b - db)], dtype=np.uint8)
    lab_upper = np.array([min(255, target_l + dl), min(255, target_a + da), min(255, target_b + db)], dtype=np.uint8)
    lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
    
    # HSV mask: hue-based matching (more robust to lighting)
    # Hue wraps around (0-180 in OpenCV), saturation and value are more lenient
    hue_range = max(15, int(color_tolerance * 0.8))
    h_lower = max(0, target_h - hue_range)
    h_upper = min(180, target_h + hue_range)
    s_lower = max(0, target_s - 50)
    s_upper = min(255, target_s + 50)
    v_lower = max(0, target_v - 60)
    v_upper = min(255, target_v + 60)
    
    hsv_mask = cv2.inRange(hsv, np.array([h_lower, s_lower, v_lower], dtype=np.uint8), 
                                np.array([h_upper, s_upper, v_upper], dtype=np.uint8))
    
    # STAGE 2: Use Lab mask primarily, supplement with HSV
    # For high tolerance, use OR; for strict tolerance, use AND
    if color_tolerance <= 15:
        combined_mask = cv2.bitwise_and(lab_mask, hsv_mask)
    else:
        # More lenient: accept colors that match either Lab or HSV
        combined_mask = cv2.bitwise_or(lab_mask, hsv_mask)
    
    # STAGE 3: Morphological refinement
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Denoise: remove noise and fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
    
    # STAGE 4: Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.info(f'Ultimate accuracy: No contours found (Lab range: L[{target_l}]±{dl}, a[{target_a}]±{da}, b[{target_b}]±{db})')
        return None, None, 0, 0.0
    
    # STAGE 5: Multi-criteria validation
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_pixels:
            continue
        
        # Circularity check (reject artifacts)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.15:
                continue
        
        # Fit ellipse for shape analysis
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                if minor_axis > 0:
                    eccentricity = major_axis / minor_axis
                    # Reject highly elongated shapes
                    if eccentricity > 4.0:
                        continue
            except:
                pass
        
        valid_contours.append((cnt, area))
    
    if not valid_contours:
        logger.info(f'Ultimate accuracy: No valid contours after filtering')
        return None, None, 0, 0.0
    
    # STAGE 6: Select best contour (largest and most central)
    largest_contour, area = max(valid_contours, key=lambda x: x[1])
    
    # STAGE 7: Refine centroid using weighted pixel colors
    M = cv2.moments(largest_contour)
    if M['m00'] <= 0:
        return None, None, 0, 0.0
    
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    
    # STAGE 8: Sub-pixel refinement (calculate color-weighted centroid)
    mask_single = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_single, [largest_contour], 0, 255, -1)
    
    # Weight pixels by color similarity score
    pixel_coords = np.argwhere(mask_single > 0)
    if len(pixel_coords) > 0:
        color_scores = np.zeros(len(pixel_coords))
        for idx, (y, x) in enumerate(pixel_coords):
            lab_pix = lab[y, x]
            lab_dist = np.sqrt((lab_pix[0] - target_l)**2 + (lab_pix[1] - target_a)**2 + (lab_pix[2] - target_b)**2)
            color_scores[idx] = max(0, 1.0 - lab_dist / 100.0)
        
        if np.sum(color_scores) > 0:
            cx = np.sum(pixel_coords[:, 1] * color_scores) / np.sum(color_scores)
            cy = np.sum(pixel_coords[:, 0] * color_scores) / np.sum(color_scores)
    
    # STAGE 9: Calculate confidence (area, position, color uniformity)
    max_area = h * w
    area_conf = min(0.95, area / (max_area * 0.15))
    
    # Check color uniformity within detected region
    if len(pixel_coords) > 0:
        color_scores_mean = np.mean(color_scores) if np.sum(color_scores) > 0 else 0.3
        color_conf = min(0.95, color_scores_mean)
    else:
        color_conf = 0.5
    
    confidence = (area_conf * 0.4 + color_conf * 0.6)
    confidence = max(0.3, min(0.95, confidence))
    
    # Flip Y-axis: Y extends upward from bottom
    cy = h - cy
    
    logger.info(f'Ultimate accuracy: centroid=({cx:.1f},{cy:.1f}), area={area}, color_conf={color_conf:.3f}, final_conf={confidence:.3f}')
    
    return float(cx), float(cy), int(area), float(confidence)


def detect_color_opencv_hsv(img, target_rgb, color_tolerance=20, min_pixels=80):
    """High-accuracy color detection using Lab color space with strict matching.
    Uses multi-stage filtering: strict color matching -> morphology -> component validation.
    Returns (cx, cy, pixel_count, confidence) or (None, None, 0, 0.0).
    """
    import cv2
    import numpy as np
    
    # Convert image to Lab color space (more perceptually uniform than HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Convert target RGB to Lab
    target_bgr = np.uint8([[[target_rgb[2], target_rgb[1], target_rgb[0]]]])
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0, 0]
    target_l, target_a, target_b = target_lab
    
    # STRICT tolerance mapping for high accuracy
    # Tolerance 20 gives tighter Lab delta ranges
    if color_tolerance <= 10:
        dl, da, db = 8, 12, 12      # Very strict
    elif color_tolerance <= 20:
        dl, da, db = 12, 15, 15     # Strict (default)
    elif color_tolerance <= 30:
        dl, da, db = 15, 20, 20     # Moderate
    else:
        dl, da, db = 20, 25, 25     # Loose
    
    # Create Lab range
    lower = np.array([max(0, target_l - dl), max(0, target_a - da), max(0, target_b - db)], dtype=np.uint8)
    upper = np.array([min(255, target_l + dl), min(255, target_a + da), min(255, target_b + db)], dtype=np.uint8)
    
    logger.info(f'Target RGB: {target_rgb}, Lab: ({target_l}, {target_a}, {target_b})')
    logger.info(f'Tolerance {color_tolerance}: Lab deltas L±{dl}, a±{da}, b±{db}')
    logger.info(f'Lab range: [{lower[0]},{lower[1]},{lower[2]}] to [{upper[0]},{upper[1]},{upper[2]}]')
    
    # STAGE 1: Color range mask
    mask = cv2.inRange(lab, lower, upper)
    
    # STAGE 2: Advanced morphological operations for high accuracy
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove small noise, close holes, clean edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask = cv2.dilate(mask, kernel_small, iterations=1)
    
    # STAGE 3: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.info('No contours found')
        return None, None, 0, 0.0
    
    # STAGE 4: Validate and select best contour
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_pixels:
            continue
        
        # Check shape validity (reject very thin/elongated artifacts)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            # Circularity metric: 4*pi*area / perimeter^2 (1.0 for circle, <0.2 for lines)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.15:  # Reject very elongated shapes
                continue
        
        valid_contours.append((cnt, area))
    
    if not valid_contours:
        logger.info(f'No valid contours (min_pixels={min_pixels})')
        return None, None, 0, 0.0
    
    # Select largest valid contour
    largest_contour, area = max(valid_contours, key=lambda x: x[1])
    
    # STAGE 5: Calculate centroid with high precision
    M = cv2.moments(largest_contour)
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        
        # Confidence: based on area
        h_img, w_img = img.shape[:2]
        max_area = h_img * w_img
        confidence = min(0.95, max(0.3, area / (max_area * 0.1)))
        
        # Flip Y-axis: Y extends upward from bottom
        cy = h_img - cy
        
        logger.info(f'Detection: centroid=({cx:.1f},{cy:.1f}), area={area}, confidence={confidence:.2f}')
        return float(cx), float(cy), int(area), float(confidence)
    
    return None, None, 0, 0.0


def detect_color_multi_threshold_opencv(img, target_rgb, color_tolerance=20, min_pixels=80):
    """Multi-threshold detection with confidence-weighted centroid averaging.
    Runs detection at multiple tolerance levels and averages results weighted by confidence.
    Falls back to iterative tolerance increase if no detection found.
    Returns (cx, cy, pixel_count, confidence) or (None, None, 0, 0.0).
    """
    import cv2
    import numpy as np
    
    logger.info(f'Starting multi-threshold detection for RGB {target_rgb}')
    
    # Try three tolerance levels: base-2, base, base+2
    tolerance_levels = [max(10, color_tolerance - 2), color_tolerance, color_tolerance + 2]
    detections = []
    
    # STAGE 1: Multi-threshold detection
    for tol in tolerance_levels:
        cx, cy, area, conf = detect_color_opencv_hsv(img, target_rgb, tol, min_pixels)
        if cx is not None and conf >= 0.25:
            detections.append((cx, cy, area, conf))
            logger.info(f'  Tolerance {tol}: centroid=({cx:.1f},{cy:.1f}), confidence={conf:.2f}')
    
    # STAGE 2: If multi-threshold found results, average them
    if detections:
        total_conf = sum(d[3] for d in detections)
        avg_cx = sum(d[0] * d[3] for d in detections) / total_conf
        avg_cy = sum(d[1] * d[3] for d in detections) / total_conf
        avg_area = int(sum(d[2] for d in detections) / len(detections))
        avg_conf = total_conf / len(detections)
        
        logger.info(f'Multi-threshold result: ({avg_cx:.1f},{avg_cy:.1f}), area={avg_area}, confidence={avg_conf:.2f}')
        return float(avg_cx), float(avg_cy), avg_area, float(avg_conf)
    
    # STAGE 3: Iterative fallback - progressively increase tolerance
    logger.info(f'No multi-threshold match found; trying iterative tolerance increase')
    for tol in range(color_tolerance + 3, 40, 2):
        cx, cy, area, conf = detect_color_opencv_hsv(img, target_rgb, tol, min_pixels)
        if cx is not None and conf >= 0.25:
            logger.info(f'  Iterative fallback at tolerance {tol}: centroid=({cx:.1f},{cy:.1f}), confidence={conf:.2f}')
            return float(cx), float(cy), int(area), float(conf)
    
    logger.info('No detection found at any tolerance level')
    return None, None, 0, 0.0


def detect_color_pillow_coords(img_p, target_rgb=(255, 0, 0), tol=50, min_pixels=80):
    """High-accuracy Pillow-based detection using strict RGB matching.
    Returns (cx, cy, pixels_in_component, confidence) or (None, None, count, 0.0).
    """
    from PIL import Image
    
    w, h = img_p.size
    pixels = img_p.load()
    matched = []
    
    t_r, t_g, t_b = target_rgb
    
    # STRICT threshold for high accuracy - no JPEG compression padding
    # For tol=20: threshold=40 (very strict), tol=50: threshold=50
    threshold = max(25, min(80, int(tol * 0.8)))
    threshold_sq = threshold * threshold
    
    logger.info(f'Pillow detection: target RGB {target_rgb}, threshold: {threshold}')
    
    # STAGE 1: Exact color matching
    for y in range(h):
        for x in range(w):
            px_val = pixels[x, y]
            # Handle different image modes
            if isinstance(px_val, int):
                r = g = b = px_val
            elif len(px_val) == 4:  # RGBA
                r, g, b = px_val[0], px_val[1], px_val[2]
            else:  # RGB
                r, g, b = px_val[0], px_val[1], px_val[2]
            
            # Euclidean distance
            dr = r - t_r
            dg = g - t_g
            db = b - t_b
            dist_sq = dr*dr + dg*dg + db*db
            
            if dist_sq <= threshold_sq:
                matched.append((x, y))
    
    # STAGE 2: Validate minimum detection size
    if len(matched) < min_pixels:
        logger.info(f'Insufficient matches: {len(matched)} < {min_pixels}')
        return None, None, len(matched), 0.0
    
    # STAGE 3: Connected components with strict validation
    matched_set = set(matched)
    visited = set()
    components = []
    neighs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for px, py in matched:
        if (px, py) in visited:
            continue
        
        # BFS for connected component
        queue = [(px, py)]
        comp = []
        visited.add((px, py))
        
        while queue:
            cx0, cy0 = queue.pop(0)
            comp.append((cx0, cy0))
            
            for dx, dy in neighs:
                nx, ny = cx0 + dx, cy0 + dy
                if (nx, ny) in matched_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        # Keep only substantial components
        if len(comp) >= min_pixels * 0.5:
            components.append(comp)
    
    if not components:
        logger.info('No valid components found')
        return None, None, len(matched), 0.0
    
    # STAGE 4: Select largest component for centroid
    best_comp = max(components, key=len)
    xs = [p[0] for p in best_comp]
    ys = [p[1] for p in best_comp]
    cx = float(sum(xs) / len(xs))
    cy = float(sum(ys) / len(ys))
    confidence = float(len(best_comp)) / float(len(matched))
    
    # Flip Y-axis: Y extends upward from bottom
    cy = h - cy
    
    logger.info(f'Component: size={len(best_comp)}, centroid=({cx:.1f},{cy:.1f}), confidence={confidence:.2f}')
    return cx, cy, len(best_comp), confidence


def detect_color_multi_threshold_pillow(img_p, target_rgb=(255, 0, 0), tol=50, min_pixels=80):
    """Multi-threshold Pillow detection with confidence-weighted averaging.
    Runs detection at multiple threshold levels and averages results weighted by confidence.
    Falls back to iterative threshold increase if no detection found.
    Returns (cx, cy, pixels_in_component, confidence) or (None, None, 0, 0.0).
    """
    logger.info(f'Starting multi-threshold Pillow detection for RGB {target_rgb}')
    
    # Try three threshold levels: base-5, base, base+5
    threshold_levels = [max(25, tol - 5), tol, min(100, tol + 5)]
    detections = []
    
    # STAGE 1: Multi-threshold detection
    for threshold in threshold_levels:
        cx, cy, area, conf = detect_color_pillow_coords(img_p, target_rgb, threshold, min_pixels)
        if cx is not None and conf >= 0.25:
            detections.append((cx, cy, area, conf))
            logger.info(f'  Threshold {threshold}: centroid=({cx:.1f},{cy:.1f}), confidence={conf:.2f}')
    
    # STAGE 2: If multi-threshold found results, average them
    if detections:
        total_conf = sum(d[3] for d in detections)
        avg_cx = sum(d[0] * d[3] for d in detections) / total_conf
        avg_cy = sum(d[1] * d[3] for d in detections) / total_conf
        avg_area = int(sum(d[2] for d in detections) / len(detections))
        avg_conf = total_conf / len(detections)
        
        logger.info(f'Multi-threshold result: ({avg_cx:.1f},{avg_cy:.1f}), area={avg_area}, confidence={avg_conf:.2f}')
        return float(avg_cx), float(avg_cy), avg_area, float(avg_conf)
    
    # STAGE 3: Iterative fallback - progressively increase threshold
    logger.info(f'No multi-threshold match found; trying iterative threshold increase')
    for threshold in range(tol + 10, 100, 5):
        cx, cy, area, conf = detect_color_pillow_coords(img_p, target_rgb, threshold, min_pixels)
        if cx is not None and conf >= 0.25:
            logger.info(f'  Iterative fallback at threshold {threshold}: centroid=({cx:.1f},{cy:.1f}), confidence={conf:.2f}')
            return float(cx), float(cy), int(area), float(conf)
    
    logger.info('No detection found at any threshold level')
    return None, None, 0, 0.0

# (old detect_red_pillow_coords kept for compatibility)
def detect_red_pillow_coords(img_p, red_min=150, score_min=15, min_pixels=80):
    return detect_color_pillow_coords(img_p, target_rgb=(255, 0, 0), tol=100, min_pixels=min_pixels)



@app.route('/')
def serve_index():
    from flask import send_from_directory
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/extract_first_frame', methods=['POST'])
def extract_first_frame():
    """Extract and return the first frame from an uploaded video."""
    if 'video' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'file type not allowed'}), 400

    # Save video temporarily
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    save_name = f"{timestamp}_{filename}"
    input_path = os.path.join(UPLOAD_DIR, save_name)
    file.save(input_path)

    try:
        # Extract just the first frame
        with tempfile.TemporaryDirectory(dir=FRAMES_DIR) as temp_frames:
            out_pattern = os.path.join(temp_frames, 'frame.jpg')
            ffmpeg_exe = find_ffmpeg()
            if not ffmpeg_exe:
                return jsonify({'error': 'ffmpeg not available'}), 500
            
            # Use fps=1 and vframes=1 to get only first frame
            cmd = [ffmpeg_exe, '-y', '-i', input_path, '-vframes', '1', out_pattern]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Read the frame
            frame_path = os.path.join(temp_frames, 'frame.jpg')
            if not os.path.exists(frame_path):
                return jsonify({'error': 'frame extraction failed'}), 500
            
            with open(frame_path, 'rb') as f:
                frame_data = f.read()
            
            # Convert to base64 for transmission
            import base64
            frame_base64 = base64.b64encode(frame_data).decode('utf-8')
            
            logger.info(f'Extracted first frame from {filename}')
            return jsonify({
                'success': True,
                'frame': f'data:image/jpeg;base64,{frame_base64}'
            })
    
    except Exception as e:
        logger.error(f'Error extracting first frame: {e}')
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up uploaded video
        try:
            os.remove(input_path)
        except:
            pass


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
        track_color = request.form.get('track_color')
        used_detector = 'none'
        markers_count = 0
        detections = {}
        frame_data_list = []  # List to store frame data for CSV
        frame_number = 0  # Counter for frame numbering
        
        # Only track if track_color is provided
        if track_color:
            logger.info(f'Received track_color: "{track_color}" (type: {type(track_color).__name__}, len: {len(track_color)})')
            # Parse color (supports named colors, #rrggbb, and 'r,g,b')
            target_rgb = parse_color_string(track_color)
            if not target_rgb:
                logger.error(f'Failed to parse color: "{track_color}"')
                return jsonify({'error': 'invalid color format', 'received': str(track_color)}), 400
            
            logger.info(f'Parsed color to RGB: {target_rgb}')
            # Get detection parameters
            try:
                min_pixels = int(request.form.get('min_pixels', 80))
                color_tolerance = int(request.form.get('color_tolerance', 20))
            except (ValueError, TypeError):
                min_pixels = 80
                color_tolerance = 20
            
            logger.info(f'Detection params: tolerance={color_tolerance}, min_pixels={min_pixels}')
            
            # For Pillow fallback: Convert to strict threshold
            # tolerance 20 -> tol=40 (very strict), tolerance 50 -> tol=50
            tol = max(25, min(80, int(color_tolerance * 0.8)))

            # Try OpenCV-based detection first (more robust); if not available, use Pillow fallback
            try:
                import cv2
                import numpy as np
                use_cv = True
                logger.info(f'Using OpenCV for color tracking: {track_color}')
            except Exception:
                use_cv = False
                logger.info(f'OpenCV not available; using Pillow fallback for color tracking: {track_color}')

            for fname in sorted(os.listdir(temp_frames)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                path = os.path.join(temp_frames, fname)
                detections[fname] = None
                frame_number += 1  # Increment frame counter
                frame_time = (frame_number - 1) * interval  # Calculate time for this frame

                if use_cv:
                    used_detector = 'simple_effective'
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # Add video pixel information at top-left corner
                    resolution_text = f"{w}x{h}"
                    cv2.putText(img, resolution_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)
                    
                    # Use SIMPLE BUT EFFECTIVE detection (fast and reliable)
                    cx, cy, area, confidence = detect_color_simple_effective(img, target_rgb, color_tolerance, min_pixels)
                    
                    # Store frame data
                    frame_x = round(cx, 1) if cx is not None else None
                    frame_y = round(cy, 1) if cy is not None else None
                    frame_data_list.append({
                        'Frame': frame_number,
                        'X': frame_x,
                        'Y': frame_y,
                        'Time': round(frame_time, 2)
                    })
                    
                    if cx is not None and cy is not None:
                        # Flip cy back to image coordinates for drawing
                        cy_draw = h - cy
                        
                        # Draw marker on the image
                        radius_px = max(6, int(min(img.shape[0], img.shape[1]) * 0.012))
                        center_int = (int(round(cx)), int(round(cy_draw)))
                        
                        # Draw circle and cross in contrasting color
                        cv2.circle(img, center_int, radius_px + 2, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                        cv2.circle(img, center_int, radius_px + 2, (0, 192, 0), thickness=1, lineType=cv2.LINE_AA)
                        cv2.drawMarker(img, center_int, (0, 128, 0), markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
                        cv2.putText(img, f"({int(round(cx))},{int(round(cy))})", (center_int[0] + 8, center_int[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                        
                        try:
                            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        except Exception:
                            cv2.imwrite(path, img)
                        
                        markers_count += 1
                        detections[fname] = {'cx': cx, 'cy': cy, 'area': area, 'confidence': confidence}
                else:
                    used_detector = 'pillow'
                    # Pillow fallback using multi-threshold detection
                    try:
                        from PIL import Image, ImageDraw
                    except Exception:
                        continue
                    
                    img_p = Image.open(path).convert('RGB')
                    w_pil, h_pil = img_p.size
                    
                    # Add video pixel information at top-left corner
                    draw_temp = ImageDraw.Draw(img_p)
                    resolution_text = f"{w_pil}x{h_pil}"
                    draw_temp.text((5, 5), resolution_text, fill='yellow')
                    
                    cx, cy, pixel_count, confidence = detect_color_multi_threshold_pillow(img_p, target_rgb, tol, min_pixels)
                    
                    # Store frame data
                    frame_x = round(cx, 1) if cx is not None else None
                    frame_y = round(cy, 1) if cy is not None else None
                    frame_data_list.append({
                        'Frame': frame_number,
                        'X': frame_x,
                        'Y': frame_y,
                        'Time': round(frame_time, 2)
                    })
                    
                    if cx is not None and cy is not None:
                        # Flip cy back to image coordinates for drawing
                        cy_draw = h_pil - cy
                        
                        # Draw marker on PIL image
                        img_pil = Image.open(path).convert('RGB')
                        draw = ImageDraw.Draw(img_pil)
                        
                        # Add video pixel information at top-left corner
                        draw.text((5, 5), resolution_text, fill='yellow')
                        
                        r = 10
                        draw.ellipse([(cx-r, cy_draw-r), (cx+r, cy_draw+r)], outline='lime', width=2)
                        draw.line([(cx-r-5, cy_draw), (cx+r+5, cy_draw)], fill='lime', width=1)
                        draw.line([(cx, cy_draw-r-5), (cx, cy_draw+r+5)], fill='lime', width=1)
                        
                        img_pil.save(path, 'JPEG', quality=95)
                        markers_count += 1
                        detections[fname] = {'cx': float(cx), 'cy': float(cy), 'area': pixel_count, 'confidence': confidence}

            # Write per-frame detections (if any) so clients can inspect results
        try:
            import json
            det_path = os.path.join(temp_frames, 'detections.json')
            with open(det_path, 'w', encoding='utf-8') as df:
                json.dump(detections, df, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Generate CSV data sheet with Frame, X, Y, Time columns
        # If no color tracking was done, still create CSV with frame numbers and times
        if not frame_data_list:
            # Create entries for all extracted frames with None values for X, Y
            extracted_frames = sorted([f for f in os.listdir(temp_frames) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for idx, fname in enumerate(extracted_frames):
                frame_number = idx + 1
                frame_time = (frame_number - 1) * interval
                frame_data_list.append({
                    'Frame': frame_number,
                    'X': None,
                    'Y': None,
                    'Time': round(frame_time, 2)
                })
        
        if frame_data_list:
            try:
                import csv
                csv_path = os.path.join(temp_frames, 'frame_data.csv')
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Frame', 'X', 'Y', 'Time']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in frame_data_list:
                        writer.writerow(row)
                logger.info(f'Generated CSV with {len(frame_data_list)} frame entries')
            except Exception as e:
                logger.error(f'Error generating CSV: {e}')

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
    # Create a short test video (1s) with a moving colored square (color from query/form), run processing, and return frames.zip
    color_param = request.form.get('track_color') or request.args.get('color') or request.args.get('track_color') or 'red'
    target_rgb = parse_color_string(color_param) or (255, 0, 0)
    
    # Get detection parameters
    try:
        min_pixels = int(request.form.get('min_pixels', 80))
        color_tolerance = int(request.form.get('color_tolerance', 20))
    except (ValueError, TypeError):
        min_pixels = 80
        color_tolerance = 20
    
    tol = max(25, min(80, int(color_tolerance * 0.8)))
    
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
        detections = {}
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
            detections[fname] = None
            
            if use_cv and target_rgb is not None:
                used_detector = 'simple_effective'
                img = cv2.imread(path)
                if img is None:
                    continue
                
                # Use SIMPLE BUT EFFECTIVE detection (fast and reliable)
                cx, cy, area, confidence = detect_color_simple_effective(img, target_rgb, color_tolerance, min_pixels)
                
                if cx is not None and cy is not None:
                    radius_px = 6
                    center_int = (int(round(cx)), int(round(cy)))
                    cv2.circle(img, center_int, radius_px + 2, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(img, center_int, radius_px + 2, (0, 192, 0), thickness=1, lineType=cv2.LINE_AA)
                    cv2.drawMarker(img, center_int, (0, 128, 0), markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
                    cv2.putText(img, f"({int(round(cx))},{int(round(cy))})", (center_int[0] + 8, center_int[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    cv2.imwrite(path, img)
                    markers_count += 1
                    detections[fname] = {'cx': cx, 'cy': cy, 'area': area, 'confidence': confidence}
            elif target_rgb is not None:
                used_detector = 'pillow'
                from PIL import Image, ImageDraw
                img_p = Image.open(path).convert('RGB')
                cx, cy, count, conf = detect_color_multi_threshold_pillow(img_p, target_rgb, tol, min_pixels)
                if cx is not None and cy is not None:
                    draw = ImageDraw.Draw(img_p)
                    radius = 6
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(0, 255, 0), width=2)
                    draw.line((cx - 8, cy, cx + 8, cy), fill=(0, 255, 0), width=1)
                    draw.line((cx, cy - 8, cx, cy + 8), fill=(0, 255, 0), width=1)
                    img_p.save(path)
                    markers_count += 1
                    detections[fname] = {'cx': float(cx), 'cy': float(cy), 'area': count, 'confidence': conf}

        # Create zip archive with frames
        zip_base = os.path.join(UPLOAD_DIR, f"demo_{int(time.time())}")
        frames_only = os.path.join(tmpd, 'frames_only')
        os.makedirs(frames_only, exist_ok=True)
        for f in os.listdir(tmpd):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(tmpd, f), os.path.join(frames_only, f))
        zip_path = shutil.make_archive(zip_base, 'zip', frames_only)
        frames_count = sum(1 for f in os.listdir(frames_only) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        logger.info('Demo produced %s frames in %s with %s markers', frames_count, tmpd, markers_count)
        
        headers = {
            'X-Frames-Count': str(frames_count),
            'X-Markers-Count': str(markers_count),
            'X-Used-Detector': used_detector,
            'X-Track-Color': color_param
        }
        resp = send_file(zip_path, as_attachment=True)
        resp.headers.update(headers)
        return resp


if __name__ == '__main__':
    app.run(debug=True)
