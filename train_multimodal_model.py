"""
Multimodal Bread Shelf Life Model Training Pipeline.
Fuses computer vision features extracted from the 50 authentic bread images
with the synchronized IoT gas sensor time-series (MQ-135 and MQ-2).
"""

import os
import cv2
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from backend.cv_analyzer import BreadVisionAnalyzer

IMAGES_DIR = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\bread_dataset\bread_dataset\bread_dataset\synth_run1\images"
SENSOR_CSV = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\sensor_data.csv"
MODEL_OUTPUT_PATH = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\backend\shelf_life_multimodal.joblib"
MODEL_LEGACY_PATH = r"C:\Users\a\Documents\Daksh\Sem3\Embedded Systems\project\backend\shelf_life_rf.joblib"

NOMINAL_TOTAL_LIFESPAN = 72.0  # Ambient shelf-life window in hours


def extract_features_and_build_dataset():
    if not os.path.exists(SENSOR_CSV):
        raise FileNotFoundError(f"Missing {SENSOR_CSV}")

    df = pd.read_csv(SENSOR_CSV)
    print(f"Loaded {len(df)} sensor records from {SENSOR_CSV}")

    # Compute elapsed and remaining hours
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_time = df["timestamp"].min()
    df["elapsed_hours"] = (df["timestamp"] - start_time).dt.total_seconds() / 3600.0
    df["remaining_hours"] = np.maximum(0.0, NOMINAL_TOTAL_LIFESPAN - df["elapsed_hours"])

    analyzer = BreadVisionAnalyzer()

    # Feature extraction lists
    mold_coverages = []
    mold_patch_counts = []
    discolorations = []
    texture_roughnesses = []
    mean_blues = []
    mean_greens = []
    mean_reds = []
    green_red_ratios = []

    print("\nExtracting Computer Vision features from dataset images...")

    for idx, row in df.iterrows():
        img_filename = row["image_file"]
        img_path = os.path.join(IMAGES_DIR, img_filename)

        if not os.path.exists(img_path):
            print(f"  [WARN] Missing image file: {img_path}")
            # Fallback zero features
            mold_coverages.append(0.0)
            mold_patch_counts.append(0)
            discolorations.append(0.0)
            texture_roughnesses.append(200.0)
            mean_blues.append(130.0)
            mean_greens.append(130.0)
            mean_reds.append(130.0)
            green_red_ratios.append(1.0)
            continue

        img = cv2.imread(img_path)
        metrics = analyzer.analyze_image(img)

        mold_coverages.append(metrics.get("mold_coverage_pct", 0.0))
        mold_patch_counts.append(metrics.get("mold_patches_count", 0))
        discolorations.append(metrics.get("discoloration_score", 0.0))
        texture_roughnesses.append(metrics.get("texture_roughness", 200.0))

        bgr = metrics.get("mean_bgr", [130.0, 130.0, 130.0])
        b, g, r = bgr[0], bgr[1], bgr[2]
        mean_blues.append(b)
        mean_greens.append(g)
        mean_reds.append(r)
        green_red_ratios.append(g / (r + 1e-5))

        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            print(f"  Processed [{idx+1}/{len(df)}] - {img_filename} (Mold: {mold_coverages[-1]}%, MQ135: {row['mq135']})")

    # Attach CV features to DataFrame
    df["mold_coverage_pct"] = mold_coverages
    df["mold_patches_count"] = mold_patch_counts
    df["discoloration_score"] = discolorations
    df["texture_roughness"] = texture_roughnesses
    df["mean_blue"] = mean_blues
    df["mean_green"] = mean_greens
    df["mean_red"] = mean_reds
    df["green_red_ratio"] = green_red_ratios

    # Engineered Gas Telemetry Features
    df["mq_ratio"] = df["mq135"] / (df["mq2"] + 1e-5)
    df["mq_product"] = df["mq135"] * df["mq2"]
    df["log_mq135"] = np.log1p(df["mq135"])
    df["log_mq2"] = np.log1p(df["mq2"])
    df["gas_severity_index"] = (df["mq135"] - 110.0) / 600.0 + (df["mq2"] - 75.0) / 500.0

    # Interaction feature: Gas degradation x Mold Coverage
    df["spoilage_interaction"] = df["gas_severity_index"] * (1.0 + df["mold_coverage_pct"] / 10.0)

    return df


def train_and_evaluate(df):
    feature_cols = [
        # IoT Gas Sensors
        "mq135",
        "mq2",
        "mq_ratio",
        "mq_product",
        "log_mq135",
        "log_mq2",
        "gas_severity_index",
        # Computer Vision Metrics
        "mold_coverage_pct",
        "mold_patches_count",
        "discoloration_score",
        "texture_roughness",
        "mean_blue",
        "mean_green",
        "mean_red",
        "green_red_ratio",
        # Cross-Modal Interaction
        "spoilage_interaction"
    ]

    X = df[feature_cols]
    y = df["remaining_hours"]

    print(f"\nTraining Dataset Matrix: {X.shape[0]} samples x {X.shape[1]} multimodal features")

    # Ensemble: Random Forest with Gradient Boosting
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_split=3,
        random_state=42
    )

    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(rf, X, y, cv=kf, scoring="r2")
    cv_mae = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error")

    print(f"\n=======================================================")
    print(f"  5-FOLD CROSS VALIDATION RESULTS")
    print(f"=======================================================")
    print(f"  Average R² Score:         {cv_r2.mean():.4f} (± {cv_r2.std():.4f})")
    print(f"  Average MAE (Hours):      {cv_mae.mean():.2f} hours (± {cv_mae.std():.2f}h)")

    # Fit full model
    rf.fit(X, y)
    y_pred = rf.predict(X)

    train_r2 = r2_score(y, y_pred)
    train_mae = mean_absolute_error(y, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"\n=======================================================")
    print(f"  FINAL MODEL TRAINING EVALUATION")
    print(f"=======================================================")
    print(f"  Training R² Score:        {train_r2:.4f}")
    print(f"  Training MAE (Hours):     {train_mae:.2f} hours")
    print(f"  Training RMSE (Hours):    {train_rmse:.2f} hours")

    # Feature Importances
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop 8 Most Influential Multimodal Features:")
    for rank, idx in enumerate(sorted_idx[:8]):
        print(f"  {rank+1}. {feature_cols[idx]:24s}: {importances[idx]*100:.2f}% importance")

    # Package model metadata
    model_artifact = {
        "model": rf,
        "feature_cols": feature_cols,
        "train_metrics": {
            "r2": float(train_r2),
            "mae": float(train_mae),
            "rmse": float(train_rmse),
            "cv_r2_mean": float(cv_r2.mean()),
            "cv_mae_mean": float(cv_mae.mean()),
        },
        "feature_importances": {feature_cols[i]: float(importances[i]) for i in sorted_idx},
        "nominal_lifespan_hours": NOMINAL_TOTAL_LIFESPAN,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_samples": len(df)
    }

    # Save to both paths
    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)
    joblib.dump(rf, MODEL_LEGACY_PATH)
    print(f"\nSaved multimodal model bundle to {MODEL_OUTPUT_PATH}")
    print(f"Updated legacy model regressor at {MODEL_LEGACY_PATH}")

    # Print Sample Predictions vs Ground Truth
    print(f"\nSample Predictions Across Lifecycle:")
    test_indices = [0, 14, 25, 35, 48]
    for tidx in test_indices:
        sample_row = df.iloc[tidx]
        pred_hours = rf.predict(sample_row[feature_cols].values.reshape(1, -1))[0]
        actual_hours = sample_row["remaining_hours"]
        print(f"  Sample #{tidx+1:02d} ({sample_row['image_file']}): Actual = {actual_hours:4.1f}h | Predicted = {pred_hours:4.1f}h (Error: {abs(pred_hours - actual_hours):.2f}h)")

    return model_artifact


if __name__ == "__main__":
    df = extract_features_and_build_dataset()
    model_artifact = train_and_evaluate(df)
