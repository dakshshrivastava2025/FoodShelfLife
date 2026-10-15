# Bread Freshness Monitoring System
Smart eNose & Computer Vision Bread Spoilage Detection

This repository contains the software, multi-modal machine learning pipeline, and live monitoring application for the AI-Based Food Freshness Monitoring & Spoilage Prediction system.

## Setup & Execution

From this directory, using the configured Python environment:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run demo_app.py
```

### System Workflow
1. **Sample Insertion:** Place the bread sample inside the measurement chamber.
2. **Reading Cycle:** Click **▶ Start Reading** to initiate the 3-minute evaluation sequence.
3. **Multi-Modal Acquisition:**
   - Gas concentrations (MQ135 and MQ2) are sampled every 5 seconds.
   - An onboard camera frame is captured in-memory for optical verification.
4. **Classification & RSL:** The feature fusion engine determines the freshness stage (Fresh, Stale, Early Mold, Heavy Mold) along with estimated Remaining Shelf Life (RSL).
5. **Reset:** Click **↺ Reset** to clear the session for a new sample.

## Machine Learning Pipeline

- **Stage A (eNose Tabular Model):** Random Forest classifier operating on multi-channel volatile dynamics ($\Delta$ change rates, rolling averages, sensor ratios, and temporal cyclic diurnal components). Test accuracy: 97.2%.
- **Stage B (Optical Feature Extraction):** Color-space segmentation computing surface fungal area and discoloration indices.
- **Stage C (Multi-Modal Feature Fusion):** Gradient Boosting fusion combining eNose gas vectors with visual features. Test accuracy: 96.1%.

To retrain the models:
```powershell
python train_model.py
```
