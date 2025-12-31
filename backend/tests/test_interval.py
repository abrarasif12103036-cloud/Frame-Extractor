import io


def test_invalid_interval_rejected(client):
    # send a small dummy file and an invalid interval
    data = {
        'video': (io.BytesIO(b'123'), 'test.mp4'),
        'interval': '-1'
    }
    rv = client.post('/upload', data=data, content_type='multipart/form-data')
    assert rv.status_code == 400
    json = rv.get_json()
    assert json is not None
    assert 'invalid interval' in json.get('error', '').lower()