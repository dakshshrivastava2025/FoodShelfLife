"""GitHub Random Forest/CV pipeline adapted to synthetic data; run 5 held out."""
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from backend.cv_analyzer import BreadVisionAnalyzer
from backend.demo_service import GAS_FEATURES, LABELS, gas_features
from backend.model import BreadShelfLifeModel

ROOT = Path(__file__).resolve().parent


def load_training_data():
    frames = []
    analyzer = BreadVisionAnalyzer()
    for run in sorted((ROOT / 'bread_dataset').glob('synth_run*')):
        df = pd.read_csv(run / 'sensor_data.csv')
        df['run'] = run.name
        # Synthetic time until generator's early-mold boundary, not a measured safety endpoint.
        crossing = df.loc[df.freshness_label >= 2, 'hours_since_start']
        if crossing.empty:
            raise ValueError(f'{run}: no early-mold endpoint')
        df['remaining_hours'] = np.maximum(0, crossing.iloc[0] - df.hours_since_start)
        df['demo_grade'] = np.select([df.spoilage_factor < .15, df.spoilage_factor < .40], [0, 1], default=2)
        features = []
        for filename in df.image_filename.fillna(''):
            if not filename:
                features.append([np.nan] * 8)
                continue
            image = cv2.imread(str(run / 'images' / filename))
            if image is None:
                raise FileNotFoundError(run / 'images' / filename)
            # Exclude the generator's printed debug class from image analysis.
            cv = analyzer.analyze_image(image[40:, :])
            b, g, r = cv['mean_bgr']
            features.append([cv['mold_coverage_pct'], cv['mold_patches_count'],
                             cv['discoloration_score'], cv['texture_roughness'],
                             b, g, r, g / (r + 1e-5)])
        visual = pd.DataFrame(features).ffill().bfill().to_numpy()
        gas = gas_features(df.mq135, df.mq2)
        matrix = np.column_stack([gas, visual, gas[:, 6] * (1 + visual[:, 0] / 10)])
        for i, col in enumerate(BreadShelfLifeModel.FEATURE_COLS):
            df[col] = matrix[:, i]
        frames.append(df)
        print(f'{run.name}: {len(df)} synthetic rows, {df.image_filename.notna().sum()} images', flush=True)
    if len(frames) < 2:
        raise ValueError('Generate the five synthetic runs first: python generate_dataset.py')
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_training_data()
    train, test = df[df.run != 'synth_run5'], df[df.run == 'synth_run5']
    if test.empty:
        raise ValueError('Missing held-out synth_run5')
    config = dict(n_estimators=180, max_depth=12, random_state=42, n_jobs=-1)
    gas = RandomForestClassifier(**config).fit(train[GAS_FEATURES].to_numpy(), train.demo_grade)
    gas_rsl = RandomForestRegressor(**config).fit(train[GAS_FEATURES].to_numpy(), train.remaining_hours)
    cols = BreadShelfLifeModel.FEATURE_COLS
    multimodal = RandomForestRegressor(**config).fit(train[cols].to_numpy(), train.remaining_hours)
    metrics = {
        'provenance': 'synthetic only; not validated on real bread',
        'train_runs': sorted(train.run.unique().tolist()), 'test_run': 'synth_run5',
        'train_rows': len(train), 'test_rows': len(test),
        'gas_grade_accuracy': accuracy_score(test.demo_grade, gas.predict(test[GAS_FEATURES].to_numpy())),
        'gas_rsl_mae_hours': mean_absolute_error(test.remaining_hours, gas_rsl.predict(test[GAS_FEATURES].to_numpy())),
        'multimodal_rsl_mae_hours': mean_absolute_error(test.remaining_hours, multimodal.predict(test[cols].to_numpy())),
        'multimodal_rsl_r2': r2_score(test.remaining_hours, multimodal.predict(test[cols].to_numpy())),
    }
    (ROOT / 'models').mkdir(exist_ok=True)
    joblib.dump(dict(gas_classifier=gas, gas_regressor=gas_rsl, multimodal=multimodal,
                     gas_features=GAS_FEATURES, labels=LABELS, metrics=metrics),
                ROOT / 'models' / 'bread_demo.joblib', compress=3)
    joblib.dump(dict(model=multimodal, feature_cols=cols, provenance=metrics['provenance'], test_metrics=metrics),
                ROOT / 'backend' / 'shelf_life_multimodal.joblib', compress=3)
    (ROOT / 'models' / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
