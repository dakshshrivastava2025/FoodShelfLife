"""
train_model.py
==============
Trains a bread freshness classifier on multi-run spoilage datasets.

Pipeline:
  1. Load all run CSVs and merge
  2. Feature engineering (rolling stats, ratio, hour-of-day, etc.)
  3. Stage A: Gas-only Random Forest -> 4-class freshness label
  4. Stage B: Image-only model -> mold-area % via colour thresholding (classical)
  5. Stage C: Fusion Gradient Boosting on sensor + image features
  6. Evaluate on held-out test run (Run 5), print confusion matrix + report
  7. Save models to bread_dataset/models/
"""

import os
import glob
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "bread_dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

LABEL_NAMES = ["Fresh", "Stale", "Early Mold", "Heavy Mold"]

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading bread freshness datasets")
print("=" * 60)

run_dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "synth_run*")))
if not run_dirs:
    raise FileNotFoundError(
        "No training run folders found!"
    )

dfs = []
for rd in run_dirs:
    csv_path = os.path.join(rd, "sensor_data.csv")
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["run"] = os.path.basename(rd)
    df["image_dir"] = os.path.join(rd, "images")
    dfs.append(df)
    print(f"  {os.path.basename(rd)}: {len(df)} rows, "
          f"labels={df['freshness_label'].value_counts().to_dict()}")

full_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(full_df)}")

# ── 2. Feature Engineering (gas) ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Feature engineering")
print("=" * 60)

def build_gas_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp")
    df["mq135_roll6"] = df.groupby("run")["mq135"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    df["mq2_roll6"] = df.groupby("run")["mq2"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    df["mq135_delta"] = df.groupby("run")["mq135"].transform(
        lambda x: x.diff(12).fillna(0)
    )
    df["mq2_delta"] = df.groupby("run")["mq2"].transform(
        lambda x: x.diff(12).fillna(0)
    )
    df["ratio"] = df["mq135"] / (df["mq2"] + 1e-3)
    df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["sin_hour"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    return df

full_df = build_gas_features(full_df)
print("  Gas feature columns added.")

GAS_FEATURES = [
    "mq135", "mq2", "mq135_roll6", "mq2_roll6",
    "mq135_delta", "mq2_delta", "ratio",
    "hours_since_start", "sin_hour", "cos_hour"
]

# ── 3. Stage A: Gas-only model ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Stage A — Gas-only Random Forest")
print("=" * 60)

train_df = full_df[full_df["run"] != "synth_run5"]
test_df  = full_df[full_df["run"] == "synth_run5"]

X_train_gas = train_df[GAS_FEATURES].values
y_train     = train_df["freshness_label"].values
X_test_gas  = test_df[GAS_FEATURES].values
y_test      = test_df["freshness_label"].values

gas_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=12,
                                   random_state=42, n_jobs=-1))
])
gas_pipe.fit(X_train_gas, y_train)
y_pred_gas = gas_pipe.predict(X_test_gas)

acc_a = accuracy_score(y_test, y_pred_gas)
print(f"  Stage A accuracy (held-out run5): {acc_a:.3f}")
print(classification_report(y_test, y_pred_gas,
                              target_names=LABEL_NAMES, zero_division=0))

joblib.dump(gas_pipe, os.path.join(MODEL_DIR, "gas_model.pkl"))
print(f"  Saved -> models/gas_model.pkl")

# ── 4. Stage B: Image-only (colour-thresholding mold area %) ─────────────────
print("\n" + "=" * 60)
print("4. Stage B — Image mold-area extraction")
print("=" * 60)

def mold_area_pct(img_path: str) -> float:
    """Return fraction of pixels that are mold-coloured (green-ish or dark)."""
    try:
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    except Exception:
        return 0.0

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # Green bias (mold) – green channel dominates, not too bright
    green_mask = (g > r + 10) & (g > b + 10) & (g < 200)
    # Grey-brown mold / dark patches
    dark_mask  = (img.mean(axis=2) < 100) & (g > 30)

    mold_mask = green_mask | dark_mask
    # Only consider central bread region (avoid background)
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    ry, rx = int(H * 0.35), int(W * 0.38)
    bread_region = np.zeros((H, W), dtype=bool)
    Y, X = np.ogrid[:H, :W]
    bread_region[((Y - cy)/ry)**2 + ((X - cx)/rx)**2 <= 1] = True

    bread_pixels = bread_region.sum()
    if bread_pixels == 0:
        return 0.0
    return float((mold_mask & bread_region).sum()) / bread_pixels


def extract_img_features(df: pd.DataFrame) -> np.ndarray:
    """Extract mold % for every row that has an image (interpolate others)."""
    df = df.copy().reset_index(drop=True)
    df["mold_pct"] = np.nan

    # Ensure string columns; NaN filenames become empty string
    df["image_filename"] = df["image_filename"].fillna("").astype(str)
    df["image_dir"]      = df["image_dir"].fillna("").astype(str)

    # Only rows with an image filename
    has_img = (df["image_filename"] != "") & (df["image_dir"] != "")
    rows_with_img = df.index[has_img].tolist()
    for idx in rows_with_img:
        fn   = df.at[idx, "image_filename"]
        idir = df.at[idx, "image_dir"]
        full_path = os.path.join(idir, fn)
        if os.path.exists(full_path):
            pct = mold_area_pct(full_path)
            df.at[idx, "mold_pct"] = pct

    # Preserve genuine zero-area images and never interpolate between runs.
    df["mold_pct"] = df.groupby("run")["mold_pct"].transform(
        lambda values: values.interpolate(method="linear").ffill().fillna(0.0)
    )
    return df["mold_pct"].values


print("  Extracting mold-area % from training images...")
train_mold = extract_img_features(train_df)
test_mold  = extract_img_features(test_df)
print(f"  Train mold_pct range: [{train_mold.min():.4f}, {train_mold.max():.4f}]")
print(f"  Test  mold_pct range: [{test_mold.min():.4f},  {test_mold.max():.4f}]")

# ── 5. Stage C: Fusion model ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Stage C — Fusion (gas + image) Gradient Boosting")
print("=" * 60)

X_train_fused = np.column_stack([X_train_gas, train_mold])
X_test_fused  = np.column_stack([X_test_gas,  test_mold])

fusion_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gb", GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                       learning_rate=0.05, random_state=42))
])
fusion_pipe.fit(X_train_fused, y_train)
y_pred_fused = fusion_pipe.predict(X_test_fused)

acc_c = accuracy_score(y_test, y_pred_fused)
print(f"  Stage C accuracy (held-out run5): {acc_c:.3f}")
print(classification_report(y_test, y_pred_fused,
                              target_names=LABEL_NAMES, zero_division=0))

joblib.dump(fusion_pipe, os.path.join(MODEL_DIR, "fusion_model.pkl"))

# Also save the feature list and scaler details for the demo
joblib.dump(GAS_FEATURES, os.path.join(MODEL_DIR, "gas_feature_names.pkl"))
print(f"  Saved -> models/fusion_model.pkl")
print(f"  Saved -> models/gas_feature_names.pkl")

# ── 6. Plots ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Generating plots")
print("=" * 60)

# 6a – Sensor vs time for one synth run
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
sample = full_df[full_df["run"] == "synth_run1"]
axes[0].plot(sample["hours_since_start"], sample["mq135"], color="#2196F3", lw=1.2)
axes[0].set_ylabel("MQ135 (CO2/Ammonia)")
axes[0].set_title("Bread Headspace Sensor Profile – Run 1")
axes[1].plot(sample["hours_since_start"], sample["mq2"], color="#FF5722", lw=1.2)
axes[1].set_ylabel("MQ2 (Alcohol/Combustible)")
axes[1].set_xlabel("Hours since start")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "sensor_timeseries.png"), dpi=120)
plt.close()
print("  Saved -> plots/sensor_timeseries.png")

# 6b – Confusion matrix for fusion model
cm = confusion_matrix(y_test, y_pred_fused)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Fusion Model Confusion Matrix  (acc={acc_c:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=120)
plt.close()
print("  Saved -> plots/confusion_matrix.png")

# 6c – Feature importance for gas model
importances = gas_pipe.named_steps["rf"].feature_importances_
fi_df = pd.DataFrame({"feature": GAS_FEATURES, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(fi_df["feature"], fi_df["importance"], color="#4CAF50")
ax.set_xlabel("Feature Importance")
ax.set_title("Gas Model — Feature Importances (Random Forest)")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=120)
plt.close()
print("  Saved -> plots/feature_importance.png")

# 6d – Spoilage factor vs predicted label scatter
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#4CAF50", "#FF9800", "#FF5722", "#9C27B0"]
for lbl in range(4):
    mask = y_pred_fused == lbl
    ax.scatter(test_df["hours_since_start"].values[mask],
               test_df["spoilage_factor"].values[mask],
               c=colors[lbl], label=LABEL_NAMES[lbl], s=8, alpha=0.6)
ax.set_xlabel("Hours since start")
ax.set_ylabel("True spoilage factor")
ax.set_title("Fusion model predictions on held-out run5")
ax.legend(markerscale=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "predictions_scatter.png"), dpi=120)
plt.close()
print("  Saved -> plots/predictions_scatter.png")

print("\n[OK] Training complete!")
print(f"   Gas-only accuracy : {acc_a:.3f}")
print(f"   Fusion accuracy   : {acc_c:.3f}")



