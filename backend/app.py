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

# Config
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'frames')
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
    try:
        res = subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, text=True)
        return jsonify({'ffmpeg': res.stdout.splitlines()[0]})
    except Exception as e:
        return jsonify({'ffmpeg': None, 'error': str(e)}), 500


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        cmd = ['ffmpeg', '-y', '-i', input_path, '-vf', fps_filter, out_pattern]
        try:
            logger.info('Running ffmpeg with interval=%s (fps=%s): %s', interval, fps, ' '.join(cmd))
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            return jsonify({'error': 'ffmpeg not available on server'}), 500
        except subprocess.CalledProcessError as e:
            # include ffmpeg stderr for debugging
            return jsonify({'error': 'ffmpeg failed', 'detail': e.stderr}), 500

        # Zip the frames
        zip_base = os.path.join(UPLOAD_DIR, f"frames_{timestamp}")
        zip_path = shutil.make_archive(zip_base, 'zip', temp_frames)

    # Optionally remove the uploaded video to save space
    try:
        os.remove(input_path)
    except Exception:
        pass

    return send_file(zip_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
