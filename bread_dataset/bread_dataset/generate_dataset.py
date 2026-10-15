"""
generate_dataset.py
===================
Data generator utility for calibration and multi-run bread spoilage profiles.
Models MQ135/MQ2 sensor trends and optical changes over bread decay lifecycles.
"""

import os
import random
import math
import csv
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── config ───────────────────────────────────────────────────────────────────
RUNS = 5
SPOIL_HOURS_RANGE = (90, 130)   # hour at which bread becomes "heavily moldy"
GAS_INTERVAL_MIN = 5            # sensor reading every 5 min
IMG_INTERVAL_MIN = 30           # photo every 30 min
IMG_SIZE = (640, 480)           # output image size
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_START = datetime(2026, 10, 10, 11, 0, 0)

# ── spoilage sigmoid ─────────────────────────────────────────────────────────
def spoilage_factor(hours: float, spoil_hour: float) -> float:
    """Returns 0.0 (fresh) → 1.0 (fully spoiled) using a sigmoid."""
    k = 0.08          # steepness
    mid = spoil_hour * 0.65  # inflection at ~65 % of spoil time
    return 1.0 / (1.0 + math.exp(-k * (hours - mid)))

# ── freshness label ──────────────────────────────────────────────────────────
def freshness_label(sf: float) -> int:
    if sf < 0.15:  return 0   # Fresh
    if sf < 0.40:  return 1   # Stale / early change
    if sf < 0.75:  return 2   # Visible mold starting
    return 3                  # Heavily moldy

# ── sensor response curves ──────────────────────────────────────────────
MQ135_BASE, MQ135_MAX = 70, 370
MQ2_BASE,   MQ2_MAX   = 250, 580

def simulate_mq135(hours: float, spoil_hour: float, rng_state: np.random.Generator) -> int:
    sf = spoilage_factor(hours, spoil_hour)
    diurnal = 6.0 * math.sin(2 * math.pi * hours / 24 + 1.0)
    noise = rng_state.normal(0, 5)
    val = MQ135_BASE + (MQ135_MAX - MQ135_BASE) * sf + diurnal + noise
    return int(np.clip(val, MQ135_BASE - 15, MQ135_MAX + 30))

def simulate_mq2(hours: float, spoil_hour: float, rng_state: np.random.Generator) -> int:
    sf = spoilage_factor(hours, spoil_hour)
    diurnal = 12.0 * math.sin(2 * math.pi * hours / 24 + 2.5)
    noise = rng_state.normal(0, 8)
    val = MQ2_BASE + (MQ2_MAX - MQ2_BASE) * sf + diurnal + noise
    return int(np.clip(val, MQ2_BASE - 20, MQ2_MAX + 40))

# ── image generation ─────────────────────────────────────────────────────────

def bread_base_layer(img: Image.Image, sf: float):
    """Draw bread-like base texture (golden-brown darkening with spoilage)."""
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Background plate / surface
    plate_color = (210, 190, 160)
    draw.rectangle([0, 0, W, H], fill=plate_color)

    # Bread loaf shape (ellipse, centred)
    bread_top_color = (
        max(30, int(200 - sf * 90)),   # R
        max(20, int(160 - sf * 80)),   # G
        max(10, int(80  - sf * 40)),   # B
    )
    bread_side_color = (
        max(20, int(170 - sf * 70)),
        max(10, int(120 - sf * 60)),
        max(5,  int(50  - sf * 30)),
    )

    margin_x, margin_y = 80, 100
    # Side / crust shadow
    draw.ellipse(
        [margin_x + 10, margin_y + 20, W - margin_x + 10, H - margin_y + 20],
        fill=bread_side_color
    )
    # Top of loaf
    draw.ellipse(
        [margin_x, margin_y, W - margin_x, H - margin_y],
        fill=bread_top_color
    )

    # Grain / crust texture streaks
    rng = np.random.default_rng(int(sf * 1000))
    for _ in range(120):
        x0 = int(rng.uniform(margin_x, W - margin_x))
        y0 = int(rng.uniform(margin_y, H - margin_y))
        x1 = x0 + int(rng.uniform(-20, 20))
        y1 = y0 + int(rng.uniform(-5, 5))
        alpha = int(rng.uniform(40, 100))
        r = max(0, bread_top_color[0] - 30)
        g = max(0, bread_top_color[1] - 20)
        b = max(0, bread_top_color[2] - 10)
        draw.line([x0, y0, x1, y1], fill=(r, g, b, alpha), width=1)


def add_staling(img: Image.Image, sf: float):
    """Desaturate and add surface drying effect."""
    if sf < 0.15:
        return img
    factor = 1.0 - 0.3 * min(sf / 0.4, 1.0)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def add_mold_spots(img: Image.Image, sf: float, seed: int = 0):
    """Paint mold blotches (green / grey / white) for sf > 0.35."""
    if sf < 0.35:
        return img

    rng = np.random.default_rng(seed + int(sf * 1000))
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # Number of spots scales with spoilage
    n_spots = int(rng.uniform(3, 8) * ((sf - 0.35) / 0.65) ** 1.5 * 30)
    n_spots = max(2, n_spots)

    # Mold colours: green, grey-green, white-ish
    MOLD_COLORS = [
        (60, 130, 60),
        (90, 140, 80),
        (140, 160, 110),
        (200, 210, 190),
        (50, 100, 50),
        (80, 120, 70),
    ]

    margin_x, margin_y = 100, 120

    for _ in range(n_spots):
        cx = int(rng.uniform(margin_x, W - margin_x))
        cy = int(rng.uniform(margin_y, H - margin_y))
        rx = int(rng.uniform(6, 28 * sf))
        ry = int(rng.uniform(4, 20 * sf))
        color = MOLD_COLORS[int(rng.integers(0, len(MOLD_COLORS)))]
        alpha = int(rng.uniform(120, 200))
        draw.ellipse(
            [cx - rx, cy - ry, cx + rx, cy + ry],
            fill=(*color, alpha)
        )
        # Small inner spore dots
        for _ in range(int(rng.uniform(2, 8))):
            sx = cx + int(rng.uniform(-rx, rx) * 0.6)
            sy = cy + int(rng.uniform(-ry, ry) * 0.6)
            sr = int(rng.uniform(1, 4))
            dark = tuple(max(0, c - 40) for c in color)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr],
                         fill=(*dark, 200))

    return img


def generate_bread_image(sf: float, run_idx: int, hour_idx: int) -> Image.Image:
    """Compose a full bread image for a given spoilage factor."""
    img = Image.new("RGBA", IMG_SIZE, (255, 255, 255, 255))
    bread_base_layer(img, sf)
    img = add_staling(img, sf)
    img = add_mold_spots(img, sf, seed=run_idx * 10000 + hour_idx)

    # Apply slight blur (camera blur)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    # Add a subtle overlay label showing the stage (for debugging / demo)
    draw = ImageDraw.Draw(img)
    label_names = ["Fresh", "Stale", "Early Mold", "Heavy Mold"]
    label = label_names[freshness_label(sf)]
    draw.rectangle([8, 8, 160, 34], fill=(0, 0, 0, 160))
    draw.text((12, 12), f"Stage: {label}", fill=(255, 255, 255))

    return img.convert("RGB")


# ── main generation loop ──────────────────────────────────────────────────────

def generate_run(run_idx: int):
    run_name = f"synth_run{run_idx + 1}"
    spoil_hour = random.uniform(*SPOIL_HOURS_RANGE)
    run_start = RUN_START + timedelta(days=run_idx * 1)  # stagger starts
    rng_state = np.random.default_rng(run_idx * 77 + 13)

    run_dir = os.path.join(BASE_DIR, "bread_dataset", run_name)
    img_dir = os.path.join(run_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "sensor_data.csv")
    columns = ["sample_number", "timestamp", "actual_read_time",
               "mq135", "mq2", "image_filename", "camera_index",
               "hours_since_start", "spoilage_factor", "freshness_label"]

    max_hours = spoil_hour + 10  # run a bit past full spoil
    total_samples = int(max_hours * 60 / GAS_INTERVAL_MIN)
    img_interval_samples = IMG_INTERVAL_MIN // GAS_INTERVAL_MIN

    print(f"  Run {run_idx+1}: spoil_hour={spoil_hour:.1f}h, "
          f"{total_samples} gas samples, "
          f"{total_samples // img_interval_samples} images")

    sample_number = 1
    img_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for s in range(total_samples):
            hours = s * GAS_INTERVAL_MIN / 60.0
            ts = run_start + timedelta(minutes=s * GAS_INTERVAL_MIN)
            sf = spoilage_factor(hours, spoil_hour)
            label = freshness_label(sf)

            mq135 = simulate_mq135(hours, spoil_hour, rng_state)
            mq2   = simulate_mq2  (hours, spoil_hour, rng_state)

            image_filename = ""
            if s % img_interval_samples == 0:
                fn = f"bread_{run_name}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
                img = generate_bread_image(sf, run_idx, img_count)
                img.save(os.path.join(img_dir, fn), quality=92)
                image_filename = fn
                img_count += 1

            writer.writerow([
                sample_number,
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                mq135, mq2,
                image_filename,
                1,
                round(hours, 3),
                round(sf, 4),
                label
            ])
            sample_number += 1

    print(f"    -> {csv_path}")
    print(f"    -> {img_count} images in {img_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating calibrated bread spoilage runs...")
    print("=" * 60)
    for i in range(RUNS):
        generate_run(i)
    print("\nDone. All dataset runs ready.")
