import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple, Optional


class BreadVisionAnalyzer:
    """
    Computer Vision pipeline for bread freshness, staling, and mold colony detection.
    Analyzes surface color degradation, detects mold patches, computes texture metrics,
    and produces an annotated diagnostic image.
    """

    def __init__(self):
        # Color thresholds for mold detection in HSV space
        # Greenish / Bluish fungal colonies
        self.mold_green_lower = np.array([30, 22, 20])
        self.mold_green_upper = np.array([92, 255, 190])

        # Dark / Black / Charcoal mold spots
        self.mold_dark_lower = np.array([0, 0, 0])
        self.mold_dark_upper = np.array([180, 255, 52])

    def analyze_image(self, image_data: Any) -> Dict[str, Any]:
        """
        Analyze an image (passed as filepath, bytes, or numpy ndarray).
        Returns diagnostic metrics and base64-encoded annotated visual.
        """
        img = self._load_image(image_data)
        if img is None:
            return {
                "success": False,
                "error": "Could not decode or read image",
                "mold_coverage_pct": 0.0,
                "visual_freshness_pct": 80.0,
                "discoloration_score": 0.0,
                "texture_roughness": 100.0,
                "mold_detected": False,
                "annotated_image_base64": ""
            }

        height, width = img.shape[:2]
        total_pixels = height * width

        # Convert colorspaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Texture Roughness / Staling index via Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        texture_score = min(500.0, max(20.0, laplacian_var))

        # 2. Mold Detection Masks
        # True fungal color signature (olive green, teal, blue-green with saturation)
        mask_green = cv2.inRange(hsv, np.array([30, 32, 25]), np.array([96, 255, 185]))
        
        # Localized dark mold specks (distinct from smooth lighting/shadow gradients)
        local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (27, 27))
        local_diff = local_mean - gray.astype(np.float32)
        mask_dark = ((local_diff > 32.0) & (gray < 85)).astype(np.uint8) * 255
        
        # Exclude extreme shadow border artifacts with morphology
        combined_mask = cv2.bitwise_or(mask_green, mask_dark)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of mold spots
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated_img = img.copy()
        mold_pixel_count = 0
        significant_patches = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 45:  # filter noise
                mold_pixel_count += area
                x, y, w, h = cv2.boundingRect(cnt)
                significant_patches.append((x, y, w, h, area))
                
                # Draw bounding box and alert contour
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.drawContours(annotated_img, [cnt], -1, (0, 255, 255), 1)

        # Mold percentage of bread surface
        mold_pct = float(round((mold_pixel_count / total_pixels) * 100.0, 2))
        
        # Color stats
        mean_bgr = img.mean(axis=(0, 1))
        mean_hsv = hsv.mean(axis=(0, 1))

        # Discoloration score (0 to 10 scale)
        # Based on deviation from fresh bread tone (warm golden/creamy)
        # and mold presence
        discoloration = min(10.0, round((mold_pct * 1.5) + (abs(mean_hsv[0] - 65) / 15.0), 1))
        
        # Visual freshness percentage (100 = spotless fresh, 0 = severe mold)
        if mold_pct > 5.0:
            visual_freshness = max(0.0, 40.0 - (mold_pct * 3.0))
        elif mold_pct > 1.0:
            visual_freshness = max(20.0, 75.0 - (mold_pct * 8.0))
        else:
            # Subtle degradation based on discoloration and texture
            visual_freshness = max(70.0, min(100.0, 100.0 - (discoloration * 3.0)))

        visual_freshness = float(round(visual_freshness, 1))
        mold_detected = bool(mold_pct > 0.8 or len(significant_patches) > 2)

        # Overlay diagnostic summary on annotated image
        status_text = "MOLD DETECTED" if mold_detected else "SURFACE HEALTHY"
        status_color = (0, 0, 230) if mold_detected else (0, 200, 50)
        
        # Top banner background
        cv2.rectangle(annotated_img, (10, 10), (380, 85), (20, 20, 20), -1)
        cv2.rectangle(annotated_img, (10, 10), (380, 85), status_color, 2)
        cv2.putText(annotated_img, f"CV Status: {status_text}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(annotated_img, f"Mold Ratio: {mold_pct}% | Patches: {len(significant_patches)}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(annotated_img, f"Visual Freshness: {visual_freshness}%",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 230, 255), 1)

        # Encode annotated image to base64 JPEG
        ret, buf = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64_str = base64.b64encode(buf).decode("utf-8") if ret else ""

        return {
            "success": True,
            "mold_detected": mold_detected,
            "mold_coverage_pct": mold_pct,
            "mold_patches_count": len(significant_patches),
            "visual_freshness_pct": visual_freshness,
            "discoloration_score": discoloration,
            "texture_roughness": round(texture_score, 1),
            "mean_bgr": [round(float(c), 1) for c in mean_bgr],
            "annotated_image_base64": f"data:image/jpeg;base64,{b64_str}"
        }

    def _load_image(self, image_data: Any) -> Optional[np.ndarray]:
        """Loads image from str path, bytes, or numpy array."""
        if isinstance(image_data, np.ndarray):
            return image_data
        elif isinstance(image_data, str):
            return cv2.imread(image_data)
        elif isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return None
