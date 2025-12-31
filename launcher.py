#!/usr/bin/env python3
"""Launcher for packaged executable.
Starts the Flask server in a background thread, waits for health, then opens the UI file in the default browser.
"""
import os
import sys
import threading
import time
import webbrowser

# ensure imports work when frozen
if getattr(sys, 'frozen', False):
    base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
else:
    base = os.path.dirname(__file__)

# Add repo root to path so `backend` package can be imported even when running from source
sys.path.insert(0, base)

from backend.app import app  # import after path adjustment


def run_server():
    # Run Flask app without debug and with reloader off
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    # Wait for server to report healthy
    import socket

    def wait_for_port(host, port, timeout=15.0):
        start = time.time()
        while time.time() - start < timeout:
            try:
                import urllib.request
                urllib.request.urlopen(f'http://{host}:{port}/health', timeout=1)
                return True
            except Exception:
                time.sleep(0.3)
        return False

    ok = wait_for_port('127.0.0.1', 5000, timeout=15.0)

    # Open the local frontend file for a consistent experience
    index_path = os.path.join(base, 'frontend', 'index.html')
    if os.path.exists(index_path):
        webbrowser.open('file://' + os.path.abspath(index_path))
    elif ok:
        webbrowser.open('http://127.0.0.1:5000')
    else:
        print('Server did not start within timeout. Check logs in console.')

    # Keep main thread alive while server runs
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Shutting down.')
