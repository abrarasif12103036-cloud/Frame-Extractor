import os
import shutil
import subprocess
import sys
import time
import zipfile
import requests

HOST = 'http://127.0.0.1:5000'


def generate_test_video(path):
    # create a 1 second test video at 10 fps (expect ~10 frames)
    cmd = [
        'ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=10', path
    ]
    subprocess.run(cmd, check=True)


def upload_and_get_zip(video_path, out_zip):
    with open(video_path, 'rb') as f:
        files = {'video': ('sample.mp4', f, 'video/mp4')}
        r = requests.post(f'{HOST}/upload', files=files, stream=True)
        r.raise_for_status()
        with open(out_zip, 'wb') as of:
            for chunk in r.iter_content(chunk_size=8192):
                of.write(chunk)


def count_frames_in_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        frames = [n for n in names if n.lower().endswith('.jpg')]
        return len(frames)


def main():
    video = 'sample.mp4'
    zip_out = 'frames_out.zip'
    if os.path.exists(video):
        os.remove(video)
    if os.path.exists(zip_out):
        os.remove(zip_out)

    print('Generating test video...')
    generate_test_video(video)

    # give server a moment
    time.sleep(1)

    print('Uploading and retrieving zip...')
    upload_and_get_zip(video, zip_out)

    print('Counting frames in zip...')
    n = count_frames_in_zip(zip_out)
    print(f'Found {n} frames')

    if n < 8:
        print('ERROR: too few frames', file=sys.stderr)
        sys.exit(2)
    print('E2E test passed')


if __name__ == '__main__':
    main()
