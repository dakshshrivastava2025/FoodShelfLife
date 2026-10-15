import cv2
import time
from typing import List, Dict, Any, Optional


class CameraService:
    """
    Manages physical cameras accessible via OpenCV on the host machine.
    Supports index 0 (typical laptop webcam) and index 1+ (external USB cameras).
    """

    @staticmethod
    def list_cameras(max_indices: int = 4) -> List[Dict[str, Any]]:
        """Scans and lists available cameras on the machine."""
        available = []
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        
        for idx in range(max_indices):
            opened = False
            for b in backends:
                cap = cv2.VideoCapture(idx, b)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        name = "Laptop Webcam (Default)" if idx == 0 else f"External / USB Camera {idx}"
                        available.append({
                            "index": idx,
                            "name": name,
                            "resolution": f"{w}x{h}",
                            "backend": "DSHOW" if b == cv2.CAP_DSHOW else "MSMF"
                        })
                        opened = True
                        cap.release()
                        break
                    cap.release()
        return available

    @staticmethod
    def capture_frame(camera_index: int = 0) -> Optional[Any]:
        """Captures a single frame from the specified camera index."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for b in backends:
            cap = cv2.VideoCapture(camera_index, b)
            if not cap.isOpened():
                continue
            
            # Flush a couple frames to let auto-exposure adjust
            for _ in range(4):
                cap.read()
                time.sleep(0.05)
                
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame
        return None
