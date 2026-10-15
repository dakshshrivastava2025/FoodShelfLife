import os
import glob
import pandas as pd
from typing import List, Dict, Any


class DatasetService:
    """Provides historical sensor data trends and sample gallery from the bread experiment."""

    def __init__(self, csv_path: str = "sensor_data.csv", images_dir: str = "bread_dataset/run1/images"):
        self.csv_path = csv_path
        self.images_dir = images_dir

    def get_timeline_summary(self, max_points: int = 80) -> Dict[str, Any]:
        """Returns sampled timeline of sensor readings for charting."""
        if not os.path.exists(self.csv_path):
            return {"points": [], "total_samples": 0}

        try:
            df = pd.read_csv(self.csv_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            start_time = df["timestamp"].min()
            df["elapsed_hours"] = ((df["timestamp"] - start_time).dt.total_seconds() / 3600.0).round(2)
            
            # Subsample for smooth charts
            step = max(1, len(df) // max_points)
            sampled = df.iloc[::step]

            points = []
            for _, row in sampled.iterrows():
                points.append({
                    "sample": int(row.get("sample_number", 0)),
                    "timestamp": str(row["timestamp"]),
                    "elapsed_hours": float(row["elapsed_hours"]),
                    "mq135": int(row["mq135"]),
                    "mq2": int(row["mq2"]),
                    "has_image": bool(pd.notna(row.get("image_filename")) and str(row.get("image_filename")).strip() != "")
                })

            return {
                "total_samples": len(df),
                "duration_hours": float(df["elapsed_hours"].max()),
                "points": points
            }
        except Exception as e:
            print(f"[DatasetService] Error parsing CSV: {e}")
            return {"points": [], "total_samples": 0}

    def get_sample_images(self) -> List[Dict[str, str]]:
        """Returns list of reference bread images available in the project."""
        samples = []
        # Check bread_dataset folders and workspace root
        patterns = [
            os.path.join(self.images_dir, "*.jpg"),
            os.path.join("bread_dataset", "bread_dataset", "bread_dataset", "synth_run1", "images", "*.jpg"),
            os.path.join("bread_dataset", "bread_dataset", "run1", "images", "*.jpg"),
            "*.jpg.jpeg"
        ]
        
        found = []
        for p in patterns:
            found.extend(glob.glob(p))
            
        seen = set()
        for fpath in sorted(found):
            fname = os.path.basename(fpath)
            if fname in seen:
                continue
            seen.add(fname)
            samples.append({
                "filename": fname,
                "path": fpath.replace("\\", "/"),
                "url": f"/api/dataset/image/{fname}"
            })
        return samples
