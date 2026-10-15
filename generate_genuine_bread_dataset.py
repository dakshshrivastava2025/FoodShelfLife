"""
generate_genuine_bread_dataset.py
=================================
Generates genuine, photo-realistic bread sample images to replace the cartoon-like
synthetic images in synth_run1/images.

Uses the authentic macro bread crumb image (bread_run1_20261010_110000.jpg) as the ground truth
and models authentic bread degradation over a 5-day lifecycle:
  1. Fresh Phase (0-24h): Pristine porous crumb, soft natural light, moist crumb texture.
  2. Staling Phase (24-48h): Desiccation, subtle crumb yellowing, deeper contrast in pores.
  3. Early Spoilage (48-72h): Dullness, faint greyish-green micro-specks in crumb cavities.
  4. Advanced Mold Spoilage (72-120h+): Realistic fungal mycelium halos and olive/teal/charcoal
     spore colonies organically embedded in the bread's natural porous texture.
"""

import os
import glob
import random
import csv
import math
from datetime import datetime, timedelta
import cv2
import numpy as np

# Reproducibility
random.seed(42)
np.random.seed(42)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_IMG_DIR = os.path.join(PROJECT_DIR, "bread_dataset", "bread_dataset", "run1", "images")
SYNTH_IMG_DIR = os.path.join(PROJECT_DIR, "bread_dataset", "bread_dataset", "bread_dataset", "synth_run1", "images")
SYNTH_RUN_DIR = os.path.join(PROJECT_DIR, "bread_dataset", "bread_dataset", "bread_dataset", "synth_run1")

os.makedirs(SYNTH_IMG_DIR, exist_ok=True)

# Ground truth real image (first image from user)
GROUND_TRUTH_PATH = os.path.join(REAL_IMG_DIR, "bread_run1_20261010_110000.jpg")
if not os.path.exists(GROUND_TRUTH_PATH):
    # Fallback to root directory
    GROUND_TRUTH_PATH = os.path.join(PROJECT_DIR, "bread_run1_20261010_110000.jpg.jpeg")

print(f"Loading reference authentic bread photo from: {GROUND_TRUTH_PATH}")
ref_img = cv2.imread(GROUND_TRUTH_PATH)
if ref_img is None:
    raise FileNotFoundError(f"Could not load reference image at {GROUND_TRUTH_PATH}")

H, W = ref_img.shape[:2]


def add_realistic_staling(img: np.ndarray, progress: float) -> np.ndarray:
    """
    Simulates optical staling: moisture loss, crumb crystallization,
    subtle yellowing/drying, and enhanced shadow contrast in crumb pores.
    progress: 0.0 (fresh) to 1.0 (advanced stale).
    """
    img_f = img.astype(np.float32)

    # 1. Subtle warm drying tone (slight reduction in blue channel, subtle warmth)
    yellowing = np.array([-1.5 * progress, 0.5 * progress, 1.2 * progress], dtype=np.float32)
    img_f = np.clip(img_f + yellowing, 0, 255)

    # 2. Crumb pore contrast enhancement (cavities get slightly darker as moisture leaves)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pore_shadow = np.clip((140.0 - gray) / 80.0, 0, 1.0)
    contrast_mult = 1.0 - (pore_shadow * 0.12 * progress)
    for ch in range(3):
        img_f[:, :, ch] *= contrast_mult

    # 3. Micro surface desiccation (slight texture crispness)
    if progress > 0.3:
        blurred = cv2.GaussianBlur(img_f, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img_f, 1.0 + 0.25 * progress, blurred, -0.25 * progress, 0)
        img_f = sharpened

    return np.clip(img_f, 0, 255).astype(np.uint8)


def add_organic_mold_colony(img: np.ndarray, cx: int, cy: int, radius: float, maturity: float, rng_seed: int) -> np.ndarray:
    """
    Paints an organic fungal colony (Rhizopus / Penicillium bread mold) that embeds naturally
    into the crumb cavities rather than appearing as a flat cartoon shape.
    """
    rng = np.random.default_rng(rng_seed)
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Organic non-circular perimeter using multi-frequency perturbation
    angle = np.arctan2(y - cy, x - cx)
    distortion = 1.0 + 0.28 * np.sin(3 * angle + float(rng.uniform(0, 3.14))) \
                     + 0.18 * np.sin(7 * angle) \
                     + 0.12 * np.sin(11 * angle) \
                     + 0.08 * rng.normal(0, 0.15, (H, W))

    effective_dist = dist / (radius * distortion + 1e-5)

    # 1. Outer Mycelium Halo (translucent fuzzy white/grey hyphae dusting)
    mycelium_mask = np.clip(1.0 - effective_dist * 0.75, 0.0, 1.0) ** 2.2

    # 2. Inner Spore Core (velvety dark olive / teal-green / charcoal center)
    spore_mask = np.clip(1.0 - effective_dist * 1.45, 0.0, 1.0) ** 2.6

    # 3. Crumb Cavity Interaction: mold preferentially flourishes inside pores
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cavity_depth = np.clip((145.0 - gray) / 70.0, 0.0, 1.0)

    img_f = img.astype(np.float32)

    # Blend Outer Mycelium (ashy white/light grey hyphal fuzz)
    mycelium_color = np.array([168.0, 172.0, 164.0], dtype=np.float32)
    mycelium_blend = mycelium_mask * 0.42 * maturity
    for ch in range(3):
        img_f[:, :, ch] = img_f[:, :, ch] * (1.0 - mycelium_blend) + mycelium_color[ch] * mycelium_blend

    # Blend Core Spores (deep olive, blue-green, charcoal black spores)
    palette = [
        np.array([45.0, 95.0, 50.0], dtype=np.float32),   # Classic Penicillium green
        np.array([75.0, 110.0, 60.0], dtype=np.float32),  # Teal / Sage fungal core
        np.array([35.0, 75.0, 42.0], dtype=np.float32),   # Dark charcoal-green
        np.array([30.0, 55.0, 35.0], dtype=np.float32),   # Deep mature spore
        np.array([55.0, 85.0, 45.0], dtype=np.float32),   # Velvet mold
    ]
    spore_color = palette[int(rng.integers(0, len(palette)))]

    # Spores settle heavily in crumb crevices
    blend_spore = spore_mask * maturity * (0.75 + 0.25 * cavity_depth)
    blend_spore = np.clip(blend_spore, 0.0, 0.98)

    for ch in range(3):
        img_f[:, :, ch] = img_f[:, :, ch] * (1.0 - blend_spore) + spore_color[ch] * blend_spore

    # Micro-spore punctate flecks around the colony periphery
    n_flecks = int(rng.uniform(20, 55) * maturity)
    for _ in range(n_flecks):
        fx = int(cx + rng.uniform(-radius * 1.35, radius * 1.35))
        fy = int(cy + rng.uniform(-radius * 1.35, radius * 1.35))
        if 0 <= fx < W and 0 <= fy < H:
            fr = int(rng.uniform(1, 4))
            cv2.circle(img_f, (fx, fy), fr, (int(spore_color[0] * 0.7), int(spore_color[1] * 0.7), int(spore_color[2] * 0.7)), -1)

    return np.clip(img_f, 0, 255).astype(np.uint8)


def generate_genuine_dataset(target_count: int = 50):
    """
    Generates 50 genuine bread images spanning the 5-day lifecycle:
      - 0 to 24h   (Images 01-15): Fresh, pristine bread crumb (identical in style to first photo).
      - 24 to 48h  (Images 16-28): Staling, crumb drying, moisture loss.
      - 48 to 72h  (Images 29-38): Early spoilage onset, micro-discoloration.
      - 72 to 120h (Images 39-50): Realistic fungal colonies growing in crumb crevices.
    """
    print(f"\n==================================================")
    print(f"Generating {target_count} Genuine Bread Lifecycle Images")
    print(f"Destination: {SYNTH_IMG_DIR}")
    print(f"==================================================")

    # Remove old cartoon images from synth_run1/images
    old_files = glob.glob(os.path.join(SYNTH_IMG_DIR, "*.jpg"))
    for f in old_files:
        try:
            os.remove(f)
        except Exception:
            pass

    start_time = datetime(2026, 10, 10, 11, 0, 0)
    total_hours = 120.0
    hour_step = total_hours / float(target_count - 1)

    csv_rows = []
    generated_files = []

    # Pre-select realistic colony anchor points in natural crumb valleys
    colony_anchors = [
        {"cx": 340, "cy": 250, "max_r": 48},
        {"cx": 220, "cy": 180, "max_r": 36},
        {"cx": 450, "cy": 320, "max_r": 42},
        {"cx": 290, "cy": 360, "max_r": 30},
        {"cx": 480, "cy": 170, "max_r": 28},
        {"cx": 180, "cy": 310, "max_r": 34},
    ]

    for idx in range(target_count):
        elapsed_hours = round(idx * hour_step, 2)
        cur_dt = start_time + timedelta(hours=elapsed_hours)
        ts_str = cur_dt.strftime("%Y%m%d_%H%M%S")
        filename = f"bread_synth_run1_{ts_str}.jpg"
        filepath = os.path.join(SYNTH_IMG_DIR, filename)

        # Spoilage factor (sigmoid from 0.0 to 1.0)
        sf = 1.0 / (1.0 + math.exp(-0.075 * (elapsed_hours - 64.0)))

        # Start with reference genuine bread photo
        cur_img = ref_img.copy()

        # Slight natural camera illumination micro-drift
        exposure_jitter = np.random.uniform(-1.8, 1.8)
        cur_img = np.clip(cur_img.astype(np.float32) + exposure_jitter, 0, 255).astype(np.uint8)

        # 1. Apply staling / crumb moisture loss
        stale_factor = min(1.0, elapsed_hours / 50.0)
        cur_img = add_realistic_staling(cur_img, stale_factor)

        # 2. Spoilage & Fungal Mold Progression
        mold_pct = 0.0
        if sf > 0.22:  # Early microbial colonies start appearing
            # Scale colony count and size with spoilage factor
            active_colonies = min(len(colony_anchors), int(1 + (sf - 0.22) * 8.5))
            maturity = min(1.0, (sf - 0.22) / 0.68)
            radius_mult = 0.4 + 1.1 * maturity

            for c_idx in range(active_colonies):
                anchor = colony_anchors[c_idx]
                r = anchor["max_r"] * radius_mult
                cur_img = add_organic_mold_colony(
                    cur_img,
                    cx=anchor["cx"] + int(np.random.uniform(-4, 4)),
                    cy=anchor["cy"] + int(np.random.uniform(-4, 4)),
                    radius=r,
                    maturity=maturity,
                    rng_seed=idx * 100 + c_idx
                )
            mold_pct = round(active_colonies * maturity * 1.6, 2)

        # Write high-quality JPEG
        cv2.imwrite(filepath, cur_img, [cv2.IMWRITE_JPEG_QUALITY, 96])
        generated_files.append(filename)

        # MQ-135 & MQ-2 sensor readings correlated with this stage
        mq135 = int(round(68.0 + (sf * 24.0) + (math.sin(elapsed_hours / 12.0) * 1.5) + np.random.normal(0, 1.0)))
        mq2 = int(round(250.0 + (sf * 22.0) + np.random.normal(0, 2.5)))

        stage_label = "Fresh" if sf < 0.20 else ("Stale" if sf < 0.45 else ("Early Mold" if sf < 0.75 else "Heavy Mold"))
        freshness_label = 0 if sf < 0.20 else (1 if sf < 0.45 else (2 if sf < 0.75 else 3))

        csv_rows.append([
            idx + 1,
            cur_dt.strftime("%Y-%m-%d %H:%M:%S"),
            cur_dt.strftime("%Y-%m-%d %H:%M:%S"),
            mq135,
            mq2,
            filename,
            1,
            elapsed_hours,
            round(sf, 4),
            freshness_label,
            stage_label
        ])

        if idx % 10 == 0 or idx == target_count - 1:
            print(f"  [{idx+1:02d}/{target_count}] {filename} | Hour: {elapsed_hours:5.1f}h | Stage: {stage_label:<10} | MQ135: {mq135} | Mold: {mold_pct}%")

    # Write accompanying sensor_data.csv for synth_run1
    csv_path = os.path.join(SYNTH_RUN_DIR, "sensor_data.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_number", "timestamp", "actual_read_time",
            "mq135", "mq2", "image_filename", "camera_index",
            "hours_since_start", "spoilage_factor", "freshness_label", "stage_name"
        ])
        writer.writerows(csv_rows)

    print(f"\nSuccessfully generated {len(generated_files)} genuine bread images in {SYNTH_IMG_DIR}!")
    print(f"Generated paired sensor dataset at {csv_path}")


if __name__ == "__main__":
    generate_genuine_dataset(50)
