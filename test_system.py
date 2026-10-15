"""Integration checks for artifacts, scenario inference, and CSV-only acquisition."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from backend.demo_service import gas_features, load_bundle
from bread_data_collection import parse_reading
from test_demo import helpers

ROOT = Path(__file__).resolve().parent


class SystemTests(unittest.TestCase):
    def test_trained_models_cover_all_demo_scenarios(self):
        bundle = load_bundle()
        simulate = helpers()['generate_sensor_reading']
        batch, expected = [], []
        for age, label in [(18, 0), (58, 1), (74, 2)]:
            for seed in range(100):
                settled = np.array([simulate(tick, age, seed=seed) for tick in range(30, 36)])
                batch.append(gas_features(*settled.mean(axis=0))[0])
                expected.append(label)
        predictions = bundle['gas_classifier'].predict(batch)
        np.testing.assert_array_equal(predictions, expected)
        hours = bundle['gas_regressor'].predict(batch)
        self.assertTrue(np.isfinite(hours).all())
        self.assertTrue((hours >= 0).all())
        self.assertGreater(hours[:100].mean(), hours[100:200].mean())
        self.assertGreater(hours[100:200].mean(), hours[200:].mean())
        self.assertEqual(bundle['multimodal'].n_features_in_, 16)
        self.assertNotIn(bundle['metrics']['test_run'], bundle['metrics']['train_runs'])

    def test_dataset_image_references_are_complete(self):
        images, records = 0, 0
        for run in (ROOT / 'bread_dataset').glob('synth_run*'):
            df = pd.read_csv(run / 'sensor_data.csv')
            records += len(df)
            for name in df.image_filename.dropna():
                self.assertTrue((run / 'images' / name).is_file())
                images += 1
        self.assertEqual(records, 6910)
        self.assertEqual(images, 1152)

    def test_collector_appends_csv_and_never_creates_images(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'sensor_data.csv'
            command = [sys.executable, str(ROOT / 'bread_data_collection.py'), '--simulate',
                       '--samples', '3', '--interval', '.01', '--output', str(target)]
            subprocess.run(command, check=True, capture_output=True, timeout=20)
            subprocess.run(command, check=True, capture_output=True, timeout=20)
            with target.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row['sample_number']) for row in rows], list(range(1, 7)))
            self.assertTrue(all(row['image_filename'] == '' for row in rows))
            self.assertTrue(all(row['source'] == 'simulated' for row in rows))
            self.assertEqual(list(Path(directory).iterdir()), [target])
            target.write_text('wrong,schema\n1,2\n', encoding='utf-8')
            result = subprocess.run(command, capture_output=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_text(encoding='utf-8'), 'wrong,schema\n1,2\n')

    def test_serial_line_validation(self):
        self.assertEqual(parse_reading('80,250\r\n'), (80, 250))
        for invalid in ['', 'MQ135,MQ2', '12', '1,2,3', '-1,50', '1024,50', 'nan,20']:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_reading(invalid)


if __name__ == '__main__':
    unittest.main()
