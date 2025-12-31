import io
import zipfile
import shutil
from backend.app import app as flask_app


def test_demo_endpoint_creates_frames():
    if shutil.which('ffmpeg') is None:
        import pytest
        pytest.skip('ffmpeg not available; skipping demo endpoint test')

    client = flask_app.test_client()
    resp = client.post('/demo')
    assert resp.status_code == 200
    frames = int(resp.headers.get('X-Frames-Count', '0'))
    assert frames > 0
    # verify payload is a zip with at least one image file
    data = resp.data
    z = zipfile.ZipFile(io.BytesIO(data))
    names = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert len(names) == frames
