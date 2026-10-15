import os
import sys
import webbrowser
import threading
import time
import uvicorn

try:
    import serial.tools.list_ports
    ports = [p.device for p in serial.tools.list_ports.comports()]
except Exception:
    ports = []

print("=" * 60)
print("Food Shelf Life Dashboard")
print("=" * 60)
print(f"Python version : {sys.version.split()[0]}")
print(f"Detected Ports : {ports if ports else 'No physical Arduino detected (Simulation mode active)'}")
print("Server Host    : http://127.0.0.1:8000")
print("=" * 60)

def open_browser():
    time.sleep(1.2)
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, log_level="info")
