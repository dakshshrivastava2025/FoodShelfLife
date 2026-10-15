# 🍞 BreadGuard AI - Bread Shelf Life & Spoilage Monitoring Dashboard

A multimodal IoT and Computer Vision dashboard designed to predict bread shelf life and spoilage by combining **Arduino gas sensors (MQ-135 and MQ-2)** with **camera imaging (Laptop Webcam or External USB Camera)** and a trained **AI Multimodal Model**.

---

## 🌟 Key Features

1. **Dual Camera Input Selection**:
   - **💻 Laptop Webcam (Default)**: Uses the integrated laptop camera for immediate convenience.
   - **📹 External USB Camera**: Allows switching to an external camera attached to your bread enclosure or test chamber.
   - **Device Dropdown**: Automatically detects and lists all connected video devices.
   - **Live HUD Viewport**: Features targeting reticle ("Align Bread Sample") and shutter flash animation.
   - **Alternate Modes**: Supports selecting historical bread samples from the dataset or uploading custom bread photos.

2. **Arduino Serial Telemetry Integration**:
   - Connects to Arduino via serial communication (`COM5`, 9600 baud, matching `bread_data_collection.py`).
   - Real-time continuous background thread reading `mq135` and `mq2` telemetry.
   - **Dynamic Port Scanner**: Scan and select any active COM port directly from the dashboard.
   - **Realistic Simulation Fallback**: If Arduino is unplugged or busy, the system automatically runs in simulation mode using the real 70-hour experimental dataset (`sensor_data.csv`), ensuring zero interruptions during testing.

3. **Multimodal Machine Learning & Vision Model**:
   - **Machine Learning Regressor**: Random Forest model trained on the chronological 70-hour `sensor_data.csv` bread degradation dataset.
   - **Computer Vision Diagnostics (`cv_analyzer.py`)**:
     - Detects fungal and mold colonies (green, blue, dark patches) in HSV color space.
     - Calculates surface mold coverage percentage (`%`) and counts cluster patches.
     - Evaluates surface texture roughness (Laplacian variance) and discoloration index.
     - Produces an annotated diagnostic image with bounding boxes and contour highlights.
   - **Multimodal Fusion**: Correlates visual mold presence with volatile organic gas levels (`MQ-135`) and fermentation gas (`MQ-2`) to determine:
     - **Remaining Shelf Life** in hours and days (e.g. `53.5 hours (~2.2 Days)`).
     - **Freshness Score** (0% - 100%) with animated circular gauge.
     - **Status Badge** (*Freshly Baked*, *Good Condition*, *Early Staling*, *High Spoilage Risk*, *Spoiled - Mold Detected*).
     - **Actionable Food Safety Guidance** and projected expiration date/time.

4. **Scan History**:
   - Automatically archives all snapshots in `scans/` and records a detailed history log on the dashboard.

---

## 📁 Project Structure

```
project/
├── backend/
│   ├── app.py                     # FastAPI application serving web UI and API endpoints
│   ├── arduino_manager.py         # Serial manager (background thread, auto-reconnect, simulation mode)
│   ├── model.py                   # Multimodal bread shelf life model (ML + CV fusion)
│   ├── cv_analyzer.py             # Computer Vision pipeline (mold detection & annotation)
│   ├── camera_service.py          # OpenCV physical camera discovery and snapshot capture
│   ├── dataset_service.py         # Dataset timeline and sample loader
│   └── shelf_life_rf.joblib       # Pre-trained Random Forest model
├── static/
│   ├── css/
│   │   └── dashboard.css          # Glassmorphic dark dashboard theme
│   ├── js/
│   │   ├── dashboard.js           # UI controller (camera, Arduino polling, prediction rendering)
│   │   └── camera_manager.js      # WebRTC browser camera feed & device switching
│   └── index.html                 # Main dashboard interface
├── bread_dataset/                 # Extracted reference bread dataset
├── sensor_data.csv                # 70-hour experimental sensor dataset (MQ-135 & MQ-2)
├── run_dashboard.py               # One-click startup script
└── README.md
```

---

## 🚀 How to Run the Dashboard

### 1. Launch the Server
In your terminal, run:
```bash
python run_dashboard.py
```
Or directly via Uvicorn:
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The script will automatically launch your browser to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🔌 Arduino Connection Setup

1. Connect your Arduino Uno/Nano via USB.
2. In the dashboard's **"Arduino Gas Telemetry"** section:
   - Ensure the correct COM port is entered (default is `COM5`).
   - Click **"Scan Ports"** to detect connected Arduino ports.
   - Click **"Connect Arduino"**.
3. The dashboard will automatically read incoming serial lines formatted as `mq135,mq2` (e.g., `72,264`) and display live sensor gauges.
4. If Arduino is not plugged in, the dashboard seamlessly displays simulated readings from your previous experiment.

---

## 📷 Camera Operation

1. **Select Source**: Click **"Laptop Webcam"** or **"External Camera"** (or use the device dropdown).
2. **Targeting**: Align your bread slice or sample within the center targeting reticle.
3. **Capture & Analyze**: Click **"📸 Click Photo & Predict Shelf Life"**.
4. **View Diagnostics**: The dashboard displays the remaining shelf life, animated freshness ring, safety recommendation, and visual inspection overlay highlighting any mold spots.
