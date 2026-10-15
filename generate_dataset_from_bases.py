"""
Generate 50 realistic bread spoilage dataset images from 4 base AI-generated images.
Uses OpenCV augmentation to create natural-looking variations.
"""
import cv2
import numpy as np
import os
import shutil
import csv
from datetime import datetime, timedelta
import random

# Paths
ARTIFACT_DIR = r"C:\Users\a\.gemini\antigravity-ide\brain\26ce1d83-ad40-476f-99d2-5018898a9c13"
OUTPUT_DIR = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\bread_dataset\bread_dataset\bread_dataset\synth_run1\images"
SENSOR_CSV = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\sensor_data.csv"

# Base images (Day 1 fresh, Day 2 staling, Day 3 early mold, Day 4 moderate mold, Day 5 heavy mold)
BASE_IMAGES = {
    "fresh_day1": os.path.join(ARTIFACT_DIR, "bread_fresh_day1_1792060097810.jpg"),
    "staling_day2": os.path.join(ARTIFACT_DIR, "bread_staling_day2_1792060003186.jpg"),
    "early_mold_day3": os.path.join(ARTIFACT_DIR, "bread_early_mold_day3a_1792060085638.jpg"),
    "moderate_mold_day4": os.path.join(ARTIFACT_DIR, "bread_moderate_mold_day4_1792060040847.jpg"),
    "heavy_mold_day5": os.path.join(ARTIFACT_DIR, "bread_heavy_mold_day5_1792060054907.jpg"),
}

# Also include the original reference bread photo
ORIGINAL_REF = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\bread_run1_20261010_110000.jpg.jpeg"
if os.path.exists(ORIGINAL_REF):
    BASE_IMAGES["original_ref"] = ORIGINAL_REF

# Distribution: 50 images across 5 days (roughly 10 per day)
# Day 1 (0-14h): fresh_day1, original_ref  -> 10 images
# Day 2 (14-28h): staling_day2             -> 10 images
# Day 3 (28-42h): early_mold_day3          -> 10 images
# Day 4 (42-56h): moderate_mold_day4       -> 10 images
# Day 5 (56-70h): heavy_mold_day5          -> 10 images

SCHEDULE = [
    # (stage_key, count, hour_range_start, hour_range_end)
    ("fresh", 10, 0, 14),
    ("staling", 10, 14, 28),
    ("early_mold", 10, 28, 42),
    ("moderate_mold", 10, 42, 56),
    ("heavy_mold", 10, 56, 70),
]

# Map stage to base image keys
STAGE_TO_KEYS = {
    "fresh": ["fresh_day1"] + (["original_ref"] if "original_ref" in BASE_IMAGES else []),
    "staling": ["staling_day2", "fresh_day1"],
    "early_mold": ["early_mold_day3"],
    "moderate_mold": ["moderate_mold_day4"],
    "heavy_mold": ["heavy_mold_day5"],
}

TARGET_SIZE = (640, 480)  # W x H


def augment_image(img, seed):
    """Apply realistic augmentations to an image."""
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    result = img.copy()

    # 1. Random brightness/contrast adjustment
    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.randint(-15, 15)      # brightness
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    # 2. Slight color temperature shift (warmer/cooler)
    shift_b = rng.randint(-8, 8)
    shift_g = rng.randint(-5, 5)
    shift_r = rng.randint(-8, 8)
    b, g, r = cv2.split(result)
    b = cv2.add(b, shift_b)
    g = cv2.add(g, shift_g)
    r = cv2.add(r, shift_r)
    result = cv2.merge([b, g, r])

    # 3. Random crop (zoom into a sub-region, 75-95% of original)
    crop_factor = rng.uniform(0.75, 0.95)
    new_h = int(h * crop_factor)
    new_w = int(w * crop_factor)
    top = rng.randint(0, h - new_h)
    left = rng.randint(0, w - new_w)
    result = result[top:top + new_h, left:left + new_w]

    # 4. Random flip (horizontal and/or vertical)
    if rng.rand() > 0.5:
        result = cv2.flip(result, 1)  # horizontal
    if rng.rand() > 0.5:
        result = cv2.flip(result, 0)  # vertical

    # 5. Slight rotation (-10 to +10 degrees)
    angle = rng.uniform(-10, 10)
    center = (result.shape[1] // 2, result.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]),
                            borderMode=cv2.BORDER_REFLECT_101)

    # 6. Slight Gaussian blur to vary sharpness
    if rng.rand() > 0.6:
        ksize = rng.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    # 7. Add very subtle noise
    if rng.rand() > 0.5:
        noise = rng.normal(0, 3, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Resize to target
    result = cv2.resize(result, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    return result


def generate_sensor_data(hour, stage):
    """Generate realistic sensor readings for a given hour and spoilage stage."""
    rng = random.Random(int(hour * 1000))

    # MQ-135 (Air quality - increases with mold/spoilage)
    base_mq135 = {
        "fresh": 120,
        "staling": 180,
        "early_mold": 300,
        "moderate_mold": 500,
        "heavy_mold": 750,
    }
    mq135 = base_mq135[stage] + rng.gauss(0, 15)

    # MQ-2 (Gas/VOC sensor - increases with decomposition)
    base_mq2 = {
        "fresh": 80,
        "staling": 130,
        "early_mold": 220,
        "moderate_mold": 380,
        "heavy_mold": 580,
    }
    mq2 = base_mq2[stage] + rng.gauss(0, 12)

    # Temperature (slight variation, bread stored at room temp)
    temp = 25.0 + rng.gauss(0, 1.5)

    # Humidity (increases slightly with mold growth)
    base_humidity = {
        "fresh": 45,
        "staling": 50,
        "early_mold": 58,
        "moderate_mold": 65,
        "heavy_mold": 72,
    }
    humidity = base_humidity[stage] + rng.gauss(0, 3)

    return {
        "mq135": round(max(50, mq135), 1),
        "mq2": round(max(30, mq2), 1),
        "temperature": round(temp, 1),
        "humidity": round(humidity, 1),
    }


def main():
    # Verify base images exist
    available = {}
    for key, path in BASE_IMAGES.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                available[key] = img
                print(f"  [OK] {key}: {path} ({img.shape})")
            else:
                print(f"  [FAIL] Cannot read: {path}")
        else:
            print(f"  [MISS] Not found: {path}")

    if len(available) < 3:
        print("ERROR: Need at least 3 base images to generate dataset")
        return

    # Clean output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            os.remove(fp)
    print(f"\nCleaned output directory: {OUTPUT_DIR}")

    # Generate images and sensor data
    start_time = datetime(2026, 10, 10, 11, 0, 0)
    sensor_rows = []
    image_index = 0

    for stage, count, hour_start, hour_end in SCHEDULE:
        keys = STAGE_TO_KEYS.get(stage, [])
        # Filter to available keys
        valid_keys = [k for k in keys if k in available]
        if not valid_keys:
            # Fallback: use any available key
            valid_keys = list(available.keys())

        for i in range(count):
            image_index += 1
            hour = hour_start + (hour_end - hour_start) * (i / max(count - 1, 1))
            timestamp = start_time + timedelta(hours=hour)
            ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

            # Pick a base image (cycle through valid keys)
            base_key = valid_keys[i % len(valid_keys)]
            base_img = available[base_key]

            # Augment
            aug_img = augment_image(base_img, seed=image_index * 42 + i)

            # Save
            filename = f"bread_run1_{ts_str}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Generate sensor data
            sensor = generate_sensor_data(hour, stage)
            sensor_rows.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "mq135": sensor["mq135"],
                "mq2": sensor["mq2"],
                "temperature": sensor["temperature"],
                "humidity": sensor["humidity"],
                "image_file": filename,
            })

            print(f"  [{image_index:02d}/50] {filename} (stage={stage}, base={base_key})")

    # Write sensor_data.csv
    with open(SENSOR_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "mq135", "mq2", "temperature", "humidity", "image_file"])
        writer.writeheader()
        writer.writerows(sensor_rows)

    print(f"\nDone! Generated {image_index} images in {OUTPUT_DIR}")
    print(f"Updated sensor data: {SENSOR_CSV}")


if __name__ == "__main__":
    main()
