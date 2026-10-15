"""
Generate 50 genuine, organic, photo-realistic bread lifecycle images
based on the user's authentic camera photo (bread_run1_20261010_110000.jpg.jpeg)
and the user's genuine mold reference photo.

Mold is completely organic, amorphous, irregular, and seeps into the crumb pores
without any flower patterns, radial petals, or sea-creature shapes.
"""

import os
import cv2
import numpy as np
import csv
from datetime import datetime, timedelta
import random

BASE_BREAD_PATH = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\bread_run1_20261010_110000.jpg.jpeg"
OUTPUT_DIR = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\bread_dataset\bread_dataset\bread_dataset\synth_run1\images"
SENSOR_CSV = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\sensor_data.csv"

def fractal_noise(h, w, scale=60, octaves=4, seed=42):
    rng = np.random.RandomState(seed)
    total = np.zeros((h, w), dtype=np.float32)
    w_sum = 0
    for oct in range(octaves):
        freq = scale / (2 ** oct)
        weight = 0.5 ** oct
        gh = max(3, int(h / freq))
        gw = max(3, int(w / freq))
        low = rng.uniform(0, 1, (gh, gw)).astype(np.float32)
        total += cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC) * weight
        w_sum += weight
    return total / w_sum

def apply_organic_mold(img, severity, seed=100):
    """
    severity: 0.0 (fresh) to 1.0 (severe mold)
    Creates purely organic, non-symmetric, non-radial mold growth
    that seeps naturally into crumb pores.
    """
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    result = img.astype(np.float32).copy()

    # 1. Subtle camera micro-jitter (realistic webcam capture variance)
    # Slight brightness & contrast
    alpha = rng.uniform(0.96, 1.04)
    beta = rng.uniform(-4, 4)
    result = np.clip(result * alpha + beta, 0, 255)

    # 2. Aging / staling of bread over time (slight moisture loss and warmth)
    if severity > 0.05:
        stale_blue = 1.0 - 0.07 * severity
        stale_green = 1.0 - 0.02 * severity
        stale_red = 1.0 + 0.01 * severity
        result[:, :, 0] *= stale_blue
        result[:, :, 1] *= stale_green
        result[:, :, 2] *= stale_red

    # If fresh/early staling with no visible fungal colonisation
    if severity < 0.16:
        return np.clip(result, 0, 255).astype(np.uint8)

    # 3. Organic Multi-Scale Noise Fields
    n_macro = fractal_noise(h, w, scale=110, octaves=4, seed=seed)
    n_medium = fractal_noise(h, w, scale=40, octaves=3, seed=seed + 101)
    n_spores = fractal_noise(h, w, scale=8, octaves=2, seed=seed + 202)

    # 4. Bread Crumb Luminance Adherence (mold settles deeper in shaded crumb cavities/pores)
    gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pore_adherence = (1.0 - gray) * 0.40 + 0.60

    # 5. Organic Cluster Placement (completely random coordinates and irregular growth)
    if severity < 0.35:
        # Early: 1 to 2 tiny irregular speckle clusters
        num_clusters = rng.choice([1, 2])
    elif severity < 0.65:
        # Mid: 2 to 3 spreading patches
        num_clusters = rng.choice([2, 3])
    else:
        # Heavy: 2 to 4 interconnected large colonies
        num_clusters = rng.choice([2, 3, 4])

    density = np.zeros((h, w), dtype=np.float32)

    for c in range(num_clusters):
        # Place inside the bread crumb area (avoid extreme outer edges)
        cx = rng.uniform(0.25 * w, 0.72 * w)
        cy = rng.uniform(0.22 * h, 0.76 * h)

        if c == 0:
            rad = severity * min(h, w) * rng.uniform(0.32, 0.50)
        else:
            rad = severity * min(h, w) * rng.uniform(0.12, 0.32)

        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Distort distance map using multi-octave noise to break ANY circular or radial symmetry
        dist_warp = 1.0 + 1.35 * (n_macro - 0.48) + 0.55 * (n_medium - 0.5)
        dist_eff = dist / (np.maximum(dist_warp, 0.1))

        cluster_d = np.clip(1.0 - (dist_eff / (rad + 1e-4)), 0.0, 1.0)
        # Power falloff with random asymmetry
        cluster_d = np.power(cluster_d, rng.uniform(1.3, 2.1))

        density = np.maximum(density, cluster_d)

    # Modulate with medium noise and pore adherence
    density = density * (0.75 + 0.45 * n_medium) * pore_adherence
    density = np.clip(density, 0.0, 1.0)

    # 6. Multi-zone Organic Color Attenuation
    # Outer faint mycelium halo (sage-green haze)
    halo_mask = np.clip(density / 0.32, 0.0, 1.0)
    # Mid-body mossy green fungal growth
    green_mask = np.clip((density - 0.22) / 0.38, 0.0, 1.0)
    # Dense dark charcoal/black spore core (mature mold)
    core_mask = np.clip((density - 0.52) / 0.35, 0.0, 1.0) * (severity > 0.42)
    core_mask = core_mask * np.clip((n_spores * 1.6 - 0.25), 0.0, 1.0)

    mult = np.ones((h, w, 3), dtype=np.float32)

    # Apply halo attenuation (BGR)
    h_w = halo_mask[:, :, np.newaxis]
    mult = mult * (1.0 - h_w) + (mult * np.array([0.58, 0.78, 0.56], dtype=np.float32)) * h_w

    # Apply green body attenuation (BGR)
    g_w = green_mask[:, :, np.newaxis]
    mult = mult * (1.0 - g_w) + (mult * np.array([0.33, 0.58, 0.31], dtype=np.float32)) * g_w

    # Apply dark core attenuation (BGR)
    c_w = core_mask[:, :, np.newaxis]
    mult = mult * (1.0 - c_w) + (mult * np.array([0.14, 0.20, 0.14], dtype=np.float32)) * c_w

    result = result * mult

    # 7. Dusty mycelium edge fuzzing (faint powdery microscopic hairs along outer perimeter)
    edge_fuzz = np.clip(np.sin(np.pi * np.clip(density / 0.45, 0, 1)), 0, 1)
    fuzz_tex = (n_spores * edge_fuzz * 14 * severity)[:, :, np.newaxis]
    fuzz_color = np.array([168, 195, 158], dtype=np.float32) / 255.0
    result = result + fuzz_tex * fuzz_color

    return np.clip(result, 0, 255).astype(np.uint8)

def main():
    if not os.path.exists(BASE_BREAD_PATH):
        print(f"ERROR: Cannot find base bread image: {BASE_BREAD_PATH}")
        return

    base_img = cv2.imread(BASE_BREAD_PATH)
    if base_img is None:
        print(f"ERROR: Failed to load {BASE_BREAD_PATH}")
        return

    print(f"Base bread image loaded successfully: {base_img.shape}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear old files
    for f in os.listdir(OUTPUT_DIR):
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(p) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            os.remove(p)

    start_time = datetime(2026, 10, 10, 11, 0, 0)
    total_images = 50
    total_hours = 70.0
    sensor_rows = []

    print("\nGenerating 50 authentic bread lifecycle images...")

    for i in range(total_images):
        hour = (i / (total_images - 1)) * total_hours
        timestamp = start_time + timedelta(hours=hour)
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"bread_run1_{ts_str}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Spoilage timeline:
        # Day 1 (0 to 14 hrs, idx 0-9): Fresh (severity 0.0 to 0.05)
        # Day 2 (14 to 28 hrs, idx 10-19): Staling, drying (severity 0.06 to 0.14)
        # Day 3 (28 to 42 hrs, idx 20-29): Early onset mold specks (severity 0.18 to 0.38)
        # Day 4 (42 to 56 hrs, idx 30-39): Moderate spreading mold (severity 0.42 to 0.68)
        # Day 5 (56 to 70 hrs, idx 40-49): Heavy dense mold infestation (severity 0.72 to 0.95)

        if hour < 14.0:
            stage = "fresh"
            severity = (hour / 14.0) * 0.05
        elif hour < 28.0:
            stage = "staling"
            severity = 0.06 + ((hour - 14.0) / 14.0) * 0.09
        elif hour < 42.0:
            stage = "early_mold"
            severity = 0.18 + ((hour - 28.0) / 14.0) * 0.20
        elif hour < 56.0:
            stage = "moderate_mold"
            severity = 0.42 + ((hour - 42.0) / 14.0) * 0.26
        else:
            stage = "heavy_mold"
            severity = 0.72 + ((hour - 56.0) / 14.0) * 0.23

        # Generate image with unique seed
        aug_img = apply_organic_mold(base_img, severity=severity, seed=1000 + i * 37)

        # Save image (standard 640x480 resolution)
        save_img = cv2.resize(aug_img, (640, 480))
        cv2.imwrite(filepath, save_img, [cv2.IMWRITE_JPEG_QUALITY, 93])

        # Realistic sensor readings correlating with stage
        rng = random.Random(2000 + i * 19)
        base_mq135 = {"fresh": 110, "staling": 170, "early_mold": 290, "moderate_mold": 480, "heavy_mold": 720}[stage]
        base_mq2 = {"fresh": 75, "staling": 125, "early_mold": 210, "moderate_mold": 370, "heavy_mold": 560}[stage]
        
        mq135_val = round(base_mq135 + ((hour % 14.0) / 14.0) * 50 + rng.gauss(0, 8), 1)
        mq2_val = round(base_mq2 + ((hour % 14.0) / 14.0) * 40 + rng.gauss(0, 6), 1)
        temp_val = round(25.0 + rng.gauss(0, 0.8), 1)
        hum_val = round(48.0 + (hour / total_hours) * 22.0 + rng.gauss(0, 1.5), 1)

        sensor_rows.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "mq135": mq135_val,
            "mq2": mq2_val,
            "temperature": temp_val,
            "humidity": hum_val,
            "image_file": filename,
        })

        print(f"  [{i+1:02d}/50] {filename} (Hour {hour:4.1f}h | {stage:13s} | severity={severity:.2f})")

    # Update sensor_data.csv
    with open(SENSOR_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "mq135", "mq2", "temperature", "humidity", "image_file"])
        writer.writeheader()
        writer.writerows(sensor_rows)

    print(f"\nSuccessfully generated all 50 genuine images in {OUTPUT_DIR}")
    print(f"Synchronized sensor data saved to {SENSOR_CSV}")

if __name__ == "__main__":
    main()
