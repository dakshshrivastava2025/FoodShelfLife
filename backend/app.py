import os
import io
import cv2
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.arduino_manager import ArduinoManager
from backend.cv_analyzer import BreadVisionAnalyzer
from backend.model import BreadShelfLifeModel
from backend.camera_service import CameraService
from backend.dataset_service import DatasetService

# Initialize FastAPI application
app = FastAPI(
    title="Smart Bread Shelf Life AI Dashboard",
    description="Multimodal IoT & Computer Vision Dashboard for Bread Freshness & Shelf-life Estimation",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SCANS_DIR = os.path.join(BASE_DIR, "scans")
os.makedirs(SCANS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Core services
arduino_mgr = ArduinoManager(default_port="COM5", baud_rate=9600, csv_path=os.path.join(BASE_DIR, "sensor_data.csv"))
vision_analyzer = BreadVisionAnalyzer()
model_predictor = BreadShelfLifeModel(csv_dataset_path=os.path.join(BASE_DIR, "sensor_data.csv"))
dataset_svc = DatasetService(csv_path=os.path.join(BASE_DIR, "sensor_data.csv"), images_dir=os.path.join(BASE_DIR, "bread_dataset", "run1", "images"))

# In-memory scan log (persists latest 50 scans)
scan_history: List[Dict[str, Any]] = []


class ArduinoConnectRequest(BaseModel):
    port: str
    baud_rate: Optional[int] = 9600


class ArduinoManualRequest(BaseModel):
    mq135: int
    mq2: int


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main dashboard application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Dashboard index.html not yet created</h2>")


@app.get("/api/status")
async def get_system_status():
    """Returns general backend health and Arduino status."""
    arduino_data = arduino_mgr.get_latest_reading()
    return {
        "status": "healthy",
        "arduino": arduino_data,
        "model_loaded": model_predictor.rf_model is not None,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# ARDUINO & SENSORS ENDPOINTS
# ============================================================

@app.get("/api/arduino/status")
async def get_arduino_status():
    """Returns live MQ-135 and MQ-2 telemetry."""
    return arduino_mgr.get_latest_reading()


@app.get("/api/arduino/ports")
async def list_arduino_ports():
    """Scans and lists active serial COM ports."""
    return {"ports": arduino_mgr.list_ports()}


@app.post("/api/arduino/connect")
async def connect_arduino(req: ArduinoConnectRequest):
    """Attempts connection to specified COM port."""
    success = arduino_mgr.connect(port=req.port, baud_rate=req.baud_rate)
    return {
        "success": success,
        "status": arduino_mgr.get_latest_reading()
    }


@app.post("/api/arduino/manual")
async def set_manual_sensors(req: ArduinoManualRequest):
    """Overrides sensor readings for demonstration/testing."""
    arduino_mgr.set_manual_reading(req.mq135, req.mq2)
    return {
        "success": True,
        "status": arduino_mgr.get_latest_reading()
    }


# ============================================================
# CAMERA ENDPOINTS
# ============================================================

@app.get("/api/cameras")
async def list_available_cameras():
    """Lists OpenCV cameras connected to the server host."""
    cameras = CameraService.list_cameras()
    return {"cameras": cameras}


@app.post("/api/cameras/capture")
async def capture_host_camera(camera_index: int = 0):
    """Captures a frame directly from server-connected camera index."""
    frame = CameraService.capture_frame(camera_index)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Could not capture from camera index {camera_index}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"host_cam_{camera_index}_{timestamp}.jpg"
    filepath = os.path.join(SCANS_DIR, filename)
    cv2.imwrite(filepath, frame)

    return {
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "url": f"/api/scans/{filename}"
    }


# ============================================================
# MULTIMODAL INFERENCE ENDPOINT
# ============================================================

@app.post("/api/analyze")
async def analyze_bread_sample(
    image: UploadFile = File(...),
    camera_source: str = Form("Webcam / Camera"),
    mq135: Optional[int] = Form(None),
    mq2: Optional[int] = Form(None)
):
    """
    Main pipeline:
    1. Reads uploaded bread image.
    2. Runs Computer Vision analysis (fungal colonies, surface texture, discoloration).
    3. Fetches live Arduino sensor telemetry (or manual/fallback).
    4. Evaluates Multimodal Shelf Life Prediction Model.
    5. Returns comprehensive shelf life, safety status, and visual diagnostics.
    """
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image payload received")

        # 1. Computer Vision Pipeline
        cv_result = vision_analyzer.analyze_image(image_bytes)

        # 2. Sensor Readings (use request override if provided, else live Arduino manager)
        if mq135 is None or mq2 is None:
            latest_telemetry = arduino_mgr.get_latest_reading()
            final_mq135 = int(latest_telemetry.get("mq135", 72))
            final_mq2 = int(latest_telemetry.get("mq2", 262))
            sensor_source = latest_telemetry.get("mode", "simulation")
        else:
            final_mq135 = int(mq135)
            final_mq2 = int(mq2)
            sensor_source = "user_override"

        # 3. Multimodal Shelf Life Model
        prediction = model_predictor.predict(
            mq135=final_mq135,
            mq2=final_mq2,
            cv_results=cv_result
        )

        # Save scan image to archive
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{timestamp_str}.jpg"
        save_path = os.path.join(SCANS_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        # Construct result object
        result = {
            "id": f"scan_{len(scan_history) + 1}",
            "timestamp": datetime.now().isoformat(),
            "formatted_timestamp": datetime.now().strftime("%b %d, %Y - %I:%M %p"),
            "camera_source": camera_source,
            "sensor_source": sensor_source,
            "prediction": prediction,
            "cv_analysis": cv_result,
            "saved_image_url": f"/api/scans/{filename}"
        }

        # Store in recent history
        scan_history.insert(0, result)
        if len(scan_history) > 50:
            scan_history.pop()

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/scans/{filename}")
async def get_scan_image(filename: str):
    """Serves an archived scan photo."""
    path = os.path.join(SCANS_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Scan image not found")


@app.get("/api/history")
async def get_history():
    """Returns recent analysis scans."""
    return {"history": scan_history}


# ============================================================
# DATASET & REFERENCE SAMPLES
# ============================================================

@app.get("/api/dataset/timeline")
async def get_dataset_timeline():
    """Returns timeline curve of MQ135 and MQ2 from real bread experiment."""
    return dataset_svc.get_timeline_summary()


@app.get("/api/dataset/samples")
async def get_dataset_samples():
    """Returns list of reference bread images."""
    return {"samples": dataset_svc.get_sample_images()}


@app.get("/api/dataset/image/{filename}")
async def get_sample_image(filename: str):
    """Serves a reference bread sample image."""
    possible_paths = [
        os.path.join(BASE_DIR, "bread_dataset", "bread_dataset", "bread_dataset", "synth_run1", "images", filename),
        os.path.join(BASE_DIR, "bread_dataset", "bread_dataset", "run1", "images", filename),
        os.path.join(BASE_DIR, "bread_dataset", "run1", "images", filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, f"{filename}.jpeg")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return FileResponse(p)
    raise HTTPException(status_code=404, detail="Reference image not found")
