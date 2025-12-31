import requests

with open('sample.mp4','rb') as f:
    r = requests.post('http://127.0.0.1:5000/upload', files={'video':('sample.mp4', f, 'video/mp4')})
    print('status:', r.status_code)
    try:
        print('json:', r.json())
    except Exception:
        print('text:', r.text)
