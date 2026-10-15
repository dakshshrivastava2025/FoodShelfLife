# Bread freshness demonstration

One objective: train on synthetic gas readings and synthetic bread images, collect actual sensor readings to CSV without saving photos, and demonstrate trained gas inference alongside a real camera view.

The combined project lives here in `Documents/FoodShelfLife`. It uses the GitHub Random Forest feature pipeline and CV analyzer, with the local Streamlit frontend. The separate original `bread_dataset` workspace is preserved.

## Run

```powershell
cd C:\Users\Abhyudit\Documents\FoodShelfLife
python -m pip install -r requirements.txt
python run_dashboard.py
```

Open http://localhost:8501. Allow access to the chosen camera (default index 0, verified locally; choose 1 for an attached external camera). Camera frames stay in RAM. The preview refreshes every two seconds. If no camera is available, the demonstration still completes with an explicitly labeled placeholder.

The three sections of the single Run surface select these profiles:

| Click position | Scenario | Simulated storage age |
| --- | --- | --- |
| Left | Stale | 74 hours |
| Middle | Slightly stale | 58 hours |
| Right | Fresh | 18 hours |

Each run streams 36 placeholder gas readings over three minutes. The trained gas classifier uses the mean of the last six readings; the gas regressor supplies the displayed shelf-life estimate. Model class probabilities refer to the three demo grades. The camera is genuine but display-only in this mode, so clean-looking bread does not override the chosen gas profile. No live serial readings enter the demo. The UI identifies the sensor stream as simulated. Reset clears the session.

## Synthetic dataset and training

`bread_dataset/synth_run1` through `synth_run5` contain 6,910 synthetic sensor rows and 1,152 generated JPEGs. CSV image filenames refer only to generated training images. Raw ADC values are not calibrated gas concentrations. The generator's four source labels (fresh/stale/early mold/heavy mold) are distinct from the three demo grades, which are derived from synthetic spoilage-factor bands at 0.15 and 0.40.

```powershell
python generate_dataset.py          # optional: regenerate the deterministic synthetic runs
python train_multimodal_model.py    # train all three models
python -m unittest -v test_demo.py test_system.py
```

Runs 1–4 train; all of run 5 is held out. The image pipeline ignores the generator's corner label and carries the most recent synthetic image features forward within each run. All referenced images must exist. The GitHub 16-feature Random Forest regressor is trained on gas and CV features; a gas-only regressor and three-class classifier support the demo. No manually selected storage age is supplied to these predictors.

Artifacts:

- `models/bread_demo.joblib`: gas classifier, gas regressor, multimodal regressor, schemas and evaluation.
- `models/metrics.json`: measured held-out synthetic results.

The shelf-life target is time until the generator's early-mold boundary. These models are evaluated only on synthetic data; their outputs do not establish whether real bread is safe to eat. The multimodal regressor is trained and saved but the demonstration deliberately uses gas-only inference.

## CSV-only collection

```powershell
python bread_data_collection.py --port COM5 --interval 300
python bread_data_collection.py --port COM5 --preview --camera 1
python bread_data_collection_clock_aligned.py --port COM5
```

Real Arduino input must be `mq135,mq2`, each in the range 0–1023. Invalid/header lines are skipped. Hardware errors are reported; they never silently become simulated readings. CSV rows are flushed as they arrive. Default destination: `data/collected/sensor_data.csv`; use `--output` to select another file. Existing compatible files are appended, with continuing sample numbers and acquisition elapsed time (downtime is excluded). Incompatible schemas are rejected. `--samples N` stops after N new rows. Ctrl+C or Q in the optional preview stops collection and releases devices.

`image_filename` is always empty. No image directory is created and no camera image is written. Camera preview is optional; sensor-only collection needs no webcam. The clock-aligned entry point schedules the first row on the next five-minute boundary by default.

For a hardware-free collector check (explicitly tagged `source=simulated`):

```powershell
python bread_data_collection.py --simulate --samples 3 --interval 1 --output data/collected/smoke.csv
```

## Project layout

- `demo_app.py`: Streamlit frontend, scenario controls, camera preview, results and history.
- `run_dashboard.py`: frontend launcher.
- `backend/demo_service.py`: shared gas/multimodal feature schemas and model loading.
- `backend/cv_analyzer.py`: image feature extraction used during training.
- `generate_dataset.py`: reproducible synthetic sensor and image generation.
- `train_multimodal_model.py`: training and held-out evaluation for all three models.
- `bread_data_collection.py`: CSV-only sensor acquisition and optional camera preview.
- `bread_data_collection_clock_aligned.py`: scheduled collection entry point.
- `bread_dataset/synth_run1` through `synth_run5`: active training and evaluation data.
- `models/`: single trained model bundle and evaluation report.
- `test_demo.py`, `test_system.py`: frontend and integration checks.

The old web dashboard, nested duplicate project, sample captures, archived experiments, redundant training alias and duplicate model export have been removed from this branch. Earlier versions remain available in Git history.
