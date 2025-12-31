import pytest
from backend.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    rv = client.get('/health')
    assert rv.status_code == 200
    assert rv.get_json()['status'] == 'ok'


def test_upload_no_file(client):
    rv = client.post('/upload', data={})
    assert rv.status_code == 400


def test_info(client):
    rv = client.get('/info')
    # service may return 200 with ffmpeg info or 500 if ffmpeg not present
    assert rv.status_code in (200, 500)
    if rv.status_code == 200:
        assert 'ffmpeg' in rv.get_json()