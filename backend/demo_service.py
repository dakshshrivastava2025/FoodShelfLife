"""Shared gas features and artifact loading for the synthetic demonstration."""
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Fresh", "Slightly stale", "Stale"]
GAS_FEATURES = ["mq135", "mq2", "mq_ratio", "mq_product", "log_mq135",
                "log_mq2", "gas_severity_index"]


MULTIMODAL_FEATURES = ['mq135',
 'mq2',
 'mq_ratio',
 'mq_product',
 'log_mq135',
 'log_mq2',
 'gas_severity_index',
 'mold_coverage_pct',
 'mold_patches_count',
 'discoloration_score',
 'texture_roughness',
 'mean_blue',
 'mean_green',
 'mean_red',
 'green_red_ratio',
 'spoilage_interaction']

def gas_features(mq135, mq2):
    a, b = np.broadcast_arrays(np.asarray(mq135, dtype=float), np.asarray(mq2, dtype=float))
    return np.column_stack([a, b, a / (b + 1e-5), a * b,
                            np.log1p(a), np.log1p(b),
                            (a - 110) / 600 + (b - 75) / 500])


def load_bundle():
    path = ROOT / "models" / "bread_demo.joblib"
    if not path.exists():
        raise FileNotFoundError("Train the synthetic models first: python train_multimodal_model.py")
    return joblib.load(path)
