import urllib.request
import json

boundary = "BOUNDARY_12345"
img_path = r"bread_dataset\bread_dataset\bread_dataset\synth_run1\images\bread_run1_20261011_211708.jpg"
with open(img_path, "rb") as f:
    img_bytes = f.read()

parts = [
    f"--{boundary}".encode(),
    b'Content-Disposition: form-data; name="image"; filename="sample.jpg"',
    b"Content-Type: image/jpeg\r\n",
    img_bytes,
    f"--{boundary}".encode(),
    b'Content-Disposition: form-data; name="mq135"\r\n',
    b"290",
    f"--{boundary}".encode(),
    b'Content-Disposition: form-data; name="mq2"\r\n',
    b"210",
    f"--{boundary}--".encode()
]

payload = b"\r\n".join(parts)
req = urllib.request.Request("http://127.0.0.1:8000/api/analyze", data=payload)
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode())
    print("HTTP Status:", resp.status)
    pred = result["prediction"]
    cv = result["cv_analysis"]
    print("  Remaining Shelf Life Hours:", pred["remaining_shelf_life_hours"])
    print("  Freshness Score:           ", pred["freshness_score"])
    print("  Status Badge:              ", pred["status_badge"])
    print("  Visual Mold Coverage:      ", cv["mold_coverage_pct"], "%")
    print("  Recommendation:            ", pred["recommendation"])
