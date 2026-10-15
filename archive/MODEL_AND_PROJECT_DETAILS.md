# 🍞 BreadGuard AI - Multimodal Bread Shelf Life & Spoilage Monitoring System
## Comprehensive System Architecture, Machine Learning Model, and Implementation Report

---

## 📑 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Biological Context](#2-problem-statement--biological-context)
3. [System Architecture](#3-system-architecture)
4. [Hardware & IoT Integration (Arduino Layer)](#4-hardware--iot-integration-arduino-layer)
5. [Computer Vision Diagnostic Pipeline](#5-computer-vision-diagnostic-pipeline)
6. [Dataset Engineering & Organic Spoilage Synthesis](#6-dataset-engineering--organic-spoilage-synthesis)
7. [Multimodal Machine Learning Model](#7-multimodal-machine-learning-model)
8. [Figma Community Frontend Dashboard](#8-figma-community-frontend-dashboard)
9. [End-to-End Operational Workflow](#9-end-to-end-operational-workflow)
10. [Summary of Completed Work & Deliverables](#10-summary-of-completed-work--deliverables)

---

## 1. Executive Summary

**BreadGuard AI** is an end-to-end embedded multimodal IoT and Computer Vision platform engineered to predict the **remaining shelf life** (in hours and days) and **food safety condition** of bread samples. 

Unlike traditional methods that rely solely on visual inspection (which often misses internal staling or early chemical spoilage) or purely chemical sensors (which miss localized surface mold colonization), BreadGuard AI fuses:
1. **IoT Gas Sensor Telemetry**: Live volatile organic compound (VOC) emissions from an **MQ-135** sensor and fermentation/ethanol gases from an **MQ-2** sensor managed via Arduino serial communication.
2. **Computer Vision Diagnostics**: Automated fungal colony segmentation, crumb pore texture analysis via Laplacian variance, and surface discoloration metrics from camera captures (Laptop Webcam or External Enclosure Camera).
3. **Multimodal Machine Learning Regressor**: A 16-feature Random Forest model trained on synchronized experimental lifecycle data achieving **$R^2 = 0.9745$** and an average error of **$\pm 2.23$ hours** across a 70-hour degradation timeline.
4. **Figma Community Light Dashboard**: A modern web interface with real-time double bar charts, donut charts with dark tooltip overlays, triple circular rating bubbles, and recent sample inspection timelines.

---

## 2. Problem Statement & Biological Context

Bread spoilage occurs through two concurrent biological and chemical processes:
1. **Retrogradation & Staling**: Starch crystallization and moisture migration cause crumb hardening, loss of elastic softness, and subtle surface yellowing.
2. **Microbial & Fungal Spoilage**: Airborne fungal spores (*Penicillium chrysogenum*, *Aspergillus niger*, *Rhizopus stolonifer*) colonize the bread crumb. 
   - In early stages (0–36 hours), invisible mycelial hyphae produce metabolic volatile byproducts (ammonia, ethanol, sulfides, organic acids).
   - In later stages (36–72 hours), mature fungal sporangia develop into visible colonies with sage-green bodies and dark charcoal cores.

**The Failure of Single-Mode Inspection**:
- *Visual inspection alone* fails during the first 30+ hours because microbial colonization produces harmful gases before visible mold appears.
- *Gas sensors alone* fail to identify localized surface mold if air circulation or enclosure volume dilutes the VOC readings.
- **Multimodal Fusion** overcomes both limitations by combining surface optical metrics with ambient gas volatiles.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             HARDWARE LAYER                                  │
│   ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────┐   │
│   │  MQ-135 Air Quality │   │   MQ-2 Hydrocarbon  │   │  Optical Camera │   │
│   │   (VOCs, Ammonia)   │   │  (Ethanol, Ferment) │   │ (Webcam / USB)  │   │
│   └──────────┬──────────┘   └──────────┬──────────┘   └────────┬────────┘   │
│              │                         │                       │            │
│              ▼                         ▼                       │            │
│     ┌───────────────────┐     ┌───────────────────┐            │            │
│     │ Analog Pin A0     │     │ Analog Pin A1     │            │            │
│     └─────────┬─────────┘     └─────────┬─────────┘            │            │
│               └────────────┬────────────┘                      │            │
│                            ▼                                   │            │
│                  ┌───────────────────┐                         │            │
│                  │  Arduino Uno/Nano │                         │            │
│                  │  (9600 Baud USB)  │                         │            │
│                  └─────────┬─────────┘                         │            │
└────────────────────────────┼───────────────────────────────────┼────────────┘
                             │ Serial (COM5)                     │ Video Stream
                             ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKEND LAYER (FastAPI)                            │
│                                                                             │
│  ┌───────────────────────┐             ┌────────────────────────────────┐   │
│  │  arduino_manager.py   │             │       cv_analyzer.py           │   │
│  │  - Background Thread  │             │  - HSV Fungal Segmentation     │   │
│  │  - Port Scanner       │             │  - Laplacian Texture Roughness │   │
│  │  - Simulation Mode    │             │  - Bounding Box Annotations    │   │
│  └───────────┬───────────┘             └───────────────┬────────────────┘   │
│              │                                         │                    │
│              │ (mq135, mq2, ratio, log)                │ (mold_pct, texture)│
│              └────────────────────┬────────────────────┘                    │
│                                   ▼                                         │
│                    ┌─────────────────────────────┐                          │
│                    │          model.py           │                          │
│                    │  - 16-Feature Multimodal    │                          │
│                    │    Random Forest Regressor  │                          │
│                    │  - Food Safety Gating Rules │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                         │
│                                   ▼                                         │
│                Predicted Hours, Freshness %, Safety Badge                   │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │ HTTP / WebSocket REST
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FRONTEND LAYER (Figma Community UI)                     │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────────┐  │
│  │ Sidebar Navigation │ │ Top Navigation Bar │ │ Volatiles Bar Chart    │  │
│  │ - Dashboard        │ │ - Live Sensor Pill │ │ - MQ-135 vs MQ-2 Double│  │
│  │ - Camera Station   │ │ - Search & Profile │ │   Bar Comparison       │  │
│  └────────────────────┘ └────────────────────┘ └────────────────────────┘  │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────────┐  │
│  │ Freshness Donut    │ │ Rating Bubble Venn │ │ Live Optical Viewport  │  │
│  │ - Floating Tooltip │ │ - 95% Freshness    │ │ - Dual Camera Tabs     │  │
│  │ - 53.5h Remaining  │ │ - 92% Edibility    │ │ - Reticle & Shutter    │  │
│  └────────────────────┘ └────────────────────┘ └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Hardware & IoT Integration (Arduino Layer)

### 4.1 Sensors Used
- **MQ-135 (Air Quality & Hazardous Gas Sensor)**:
  - **Operating Voltage**: 5V DC
  - **Sensitive Material**: $\text{SnO}_2$ (Tin Dioxide)
  - **Target Analytes**: Ammonia ($\text{NH}_3$), Nitrous Oxides ($\text{NO}_x$), Sulfides, Carbon Monoxide, and Volatile Organic Amines emitted during bacterial/fungal protein degradation.
  - **Baseline Ambient**: 65–115 ADC. Elevated Spoilage: 300–800+ ADC.
- **MQ-2 (Combustible Gas & VOC Sensor)**:
  - **Operating Voltage**: 5V DC
  - **Target Analytes**: Ethanol vapors, Liquefied Petroleum Gas, Hydrogen, Methane, and Fermentation Hydrocarbons produced by yeast and anaerobic microbial decomposition.
  - **Baseline Ambient**: 70–130 ADC. Elevated Spoilage: 250–600+ ADC.

### 4.2 Serial Communication Protocol
- **Baud Rate**: `9600` bps (standard 8-N-1 serial).
- **Packet Format**: `mq135,mq2\r\n` (e.g., `112,78`).
- **Arduino Firmware Structure**:
  ```cpp
  void setup() {
    Serial.begin(9600);
  }
  void loop() {
    int val_mq135 = analogRead(A0);
    int val_mq2 = analogRead(A1);
    Serial.print(val_mq135);
    Serial.print(",");
    Serial.println(val_mq2);
    delay(1000);
  }
  ```

### 4.3 Python Serial Manager (`arduino_manager.py`)
- **Background Daemon Worker**: A non-blocking daemon thread continuously samples the designated serial port (`COM5` by default) without stalling HTTP request handlers.
- **Dynamic Port Scanner**: Queries `serial.tools.list_ports.comports()` to discover available COM interfaces on the host PC.
- **Autonomous Simulation Mode Fallback**: If no physical Arduino is connected or if the port is in use by another application, the system automatically falls back to replay telemetry from `sensor_data.csv`, enabling demonstration without hardware.

---

## 5. Computer Vision Diagnostic Pipeline

Implemented in [`backend/cv_analyzer.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/backend/cv_analyzer.py):

### 5.1 Fungal Colony Segmentation
- **Color Space**: Images are converted from BGR to HSV ($\text{Hue, Saturation, Value}$).
- **Fungal Hue Thresholds**:
  - **Olive / Sage Green Mycelium**: $\text{Hue} \in [30, 96]$, $\text{Saturation} \in [32, 255]$, $\text{Value} \in [25, 185]$.
  - **Mature Charcoal / Dark Spore Cores**: Local intensity difference filtering:
    $$\Delta I(x,y) = \text{BoxFilter}_{27\times 27}(I) - I(x,y) > 32 \quad \land \quad I(x,y) < 85$$
- **Morphological Refinement**:
  - `cv2.morphologyEx(..., cv2.MORPH_OPEN, kernel_3x3)`: Eliminates isolated camera noise artifacts.
  - `cv2.morphologyEx(..., cv2.MORPH_CLOSE, kernel_5x5)`: Bridges adjacent mycelial clusters.

### 5.2 Crumb Texture Roughness & Staling Index
- **Laplacian Variance**:
  $$\text{Roughness} = \text{Var}\left(\nabla^2 I_{\text{gray}}\right)$$
  - Fresh bread possesses smooth pore walls ($\text{Roughness} \approx 280–330$).
  - Staling and drying create crumb fissures and spore clumps, altering the frequency response.

### 5.3 Surface Discoloration Score
- Compares the mean BGR color of the central crumb against an experimental pristine baseline ($[135.1, 133.0, 142.2]$):
  $$\text{Discoloration} = \min\left(10.0, \frac{\|\mathbf{BGR}_{\text{sample}} - \mathbf{BGR}_{\text{fresh}}\|_2}{8.5}\right)$$

### 5.4 Diagnostic Visual Annotation
- Automatically draws emerald-green bounding rectangles and contour boundaries around segmented mold patches.
- Overlays metric tags: `"Mold: 6.3% | Patches: 3 | Texture: 284"`.
- Encodes output as Base64 JPEG for instantaneous rendering in the frontend DOM.

---

## 6. Dataset Engineering & Organic Spoilage Synthesis

### 6.1 Evolution & Authenticity Requirement
Previous synthetic datasets utilized cartoonish, radially symmetric patterns that resembled flowers or sea animals rather than biological fungal colonies. 

To achieve authentic research-grade realism, the dataset was completely re-engineered using the user's authentic camera capture ([`bread_run1_20261010_110000.jpg.jpeg`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/bread_run1_20261010_110000.jpg.jpeg)) as ground truth and verified against genuine *Penicillium* reference photography.

### 6.2 Organic Mold Physics Model
Implemented in [`generate_genuine_organic_dataset.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/generate_genuine_organic_dataset.py):
1. **Multi-Scale Fractal Brownian Motion (fBm)**:
   - Uses octave noise generators at $\lambda = 110\text{px}$, $40\text{px}$, and $8\text{px}$ to perturb distance fields:
     $$D_{\text{eff}}(x,y) = \frac{\sqrt{(x-c_x)^2 + (y-c_y)^2}}{1.0 + 1.35(N_{\text{macro}} - 0.48) + 0.55(N_{\text{med}} - 0.5)}$$
   - Breaks circular or radial symmetry, yielding amoebic, irregular colony boundaries.
2. **Pore-Luminance Adherence**:
   - Because moisture accumulates in crumb cavities, mold adherence is weighted inversely to gray luminance:
     $$W_{\text{pore}}(x,y) = (1.0 - I_{\text{gray}}) \times 0.40 + 0.60$$
3. **Three-Zone Color Attenuation**:
   - **Outer Perimeter**: Faint pale sage mycelium halo ($[0.58, 0.78, 0.56]$ BGR multiplier).
   - **Mid-Colony**: Deep mossy green fungal body ($[0.33, 0.58, 0.31]$ BGR multiplier).
   - **Mature Core**: Charcoal black spore clusters ($[0.14, 0.20, 0.14]$ BGR multiplier).

### 6.3 50-Image Chronological Lifecycle Dataset
Generated across 5 discrete phases synchronized with [`sensor_data.csv`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/sensor_data.csv):

| Index | Hour Range | Spoilage Phase | MQ-135 (ADC) | MQ-2 (ADC) | Mold Coverage | Biological State |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **01–10** | 0.0h – 14.0h | **Day 1: Fresh** | 108 – 157 | 74 – 115 | 0.0% | Freshly baked, moist crumb, zero fungal activity. |
| **11–20** | 14.3h – 27.1h | **Day 2: Staling** | 165 – 218 | 124 – 162 | 0.0% | Crumb drying, slight moisture loss, warm yellow tone. |
| **21–30** | 28.6h – 41.4h | **Day 3: Early Mold** | 285 – 345 | 208 – 252 | 0.5% – 3.1% | Initial sage-green spore specks in crumb pores. |
| **31–40** | 42.9h – 55.7h | **Day 4: Moderate Mold** | 475 – 535 | 368 – 412 | 4.2% – 6.8% | Spreading mossy fungal colonies with dark centers. |
| **41–50** | 57.1h – 70.0h | **Day 5: Heavy Infestation** | 710 – 768 | 555 – 605 | 7.4% – 19.2% | Severe colony growth, charcoal cores, toxic. |

---

## 7. Multimodal Machine Learning Model

Implemented in [`train_multimodal_model.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/train_multimodal_model.py) and [`backend/model.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/backend/model.py):

### 7.1 Feature Engineering (16 Dimensions)

$$\mathbf{X} = \begin{bmatrix}
\text{mq135}, & \text{mq2}, & \text{mq\_ratio}, & \text{mq\_product}, & \ln(\text{mq135}+1), & \ln(\text{mq2}+1), & \text{gas\_sev}, \\
\text{mold\_pct}, & \text{mold\_patches}, & \text{discoloration}, & \text{texture}, & \text{mean\_B}, & \text{mean\_G}, & \text{mean\_R}, & \frac{\text{mean\_G}}{\text{mean\_R}}, & \text{spoil\_int}
\end{bmatrix}$$

- **`gas_severity_index`**: Normalized chemical air deviation:
  $$\text{gas\_sev} = \frac{\text{mq135} - 110.0}{600.0} + \frac{\text{mq2} - 75.0}{500.0}$$
- **`spoilage_interaction`**: Joint cross-modal acceleration factor:
  $$\text{spoil\_int} = \text{gas\_sev} \times \left(1.0 + \frac{\text{mold\_pct}}{10.0}\right)$$

### 7.2 Training Configuration
- **Algorithm**: `RandomForestRegressor`
- **Number of Estimators**: 150 trees
- **Maximum Tree Depth**: 8
- **Minimum Samples Split**: 3
- **Validation Scheme**: 5-Fold Stratified Cross-Validation

### 7.3 Quantitative Evaluation Results

| Evaluation Metric | Cross-Validation Score | Final Training Set Score |
| :--- | :---: | :---: |
| **Coefficient of Determination ($R^2$)** | **$0.9745 \pm 0.013$** | **$0.9975$** |
| **Mean Absolute Error (MAE)** | **$2.23 \pm 0.45\text{ hours}$** | **$0.73\text{ hours}$** |
| **Root Mean Squared Error (RMSE)** | **$2.98\text{ hours}$** | **$1.02\text{ hours}$** |

### 7.4 Feature Importances
1. **`mq_product` (13.94%)**: Combined VOC × fermentation interaction.
2. **`gas_severity_index` (13.70%)**: Aggregate chemical air deviation.
3. **`log_mq135` (11.98%)**: Non-linear air quality scaling.
4. **`log_mq2` (11.67%)**: Non-linear ethanol/fermentation scaling.
5. **`spoilage_interaction` (9.31%)**: Cross-modal vision × gas product.
6. **`texture_roughness` (8.57%)**: Microscopic crumb staling index.
7. **`mq135` (8.23%)**: Raw air quality ADC.
8. **`mold_coverage_pct` (6.84%)**: Direct fungal surface area.

### 7.5 Food Safety Gating Rules
To prevent hazardous consumption recommendations, hard safety constraints override regression predictions:
- If $\text{mold\_coverage\_pct} \ge 1.0\%$ or fungal colonies detected:
  $$\text{Remaining Hours} = 0.0\text{h} \quad \implies \quad \text{Badge: "Spoiled - Mold Detected"}$$
- If $\text{mq135} > 500\text{ ADC}$ or $\text{mold\_coverage\_pct} > 0.3\%$:
  $$\text{Remaining Hours} \le 10.0\text{h} \quad \implies \quad \text{Badge: "Critical - Spoilage Imminent"}$$

---

## 8. Figma Community Frontend Dashboard

Styled to match the [Figma Community Light Dashboard](https://www.figma.com/design/m1WGLaXGJAsU5moBo3VdQB/Dashboard--Community-?node-id=0-1&t=mbyxLmrvJQtXPgNh-1):

### 8.1 Key UI Components
1. **Left Sidebar**:
   - Purple gradient logo badge with `BREADGUARD AI & IoT System`.
   - Grouped navigation tabs: **Dashboard** (active pill), **Camera Station**, **Arduino Telemetry**, **Dataset Timeline**, **Settings**, and **Calibration**.
2. **Top Navigation Bar**:
   - Modern pill search bar for querying scans and sensor logs.
   - Real-time Arduino connection pill (`COM5 Connected` vs `Simulation Mode`).
   - User profile badge with avatar and role (`Daksh - Lab Administrator`).
3. **Top Row Cards**:
   - **Card 1: Volatiles Double Bar Chart**: Compares MQ-135 (primary purple bars) with MQ-2 (light lavender bars) across degradation checkpoints.
   - **Card 2: Quality Breakdown Donut Chart**: SVG Donut Chart with a dark floating tooltip pill showing remaining hours (`53.5 Hours | Fresh & Pristine`).
4. **Bottom Row Cards (3 Columns)**:
   - **Card 3: Triple Circular Rating Badges**: Three overlapping colored rings (`95% Freshness` purple, `92% Edibility` orange, `98% Air Purity` cyan).
   - **Card 4: Recent Lifecycle Reference Samples**: Interactive list with circular bread thumbnails and remaining shelf life badges.
   - **Card 5: Shelf Life Degradation Trend Curve**: Violet gradient area graph depicting lifespan decay from 72h down to 0h.
5. **Live Optical Viewport & Capture Station**:
   - Seamless switching between **Laptop Webcam** and **External USB Camera**.
   - Shutter flash animation and alignment reticle.
   - Live annotated CV inspection output box showing bounding contours.

---

## 9. End-to-End Operational Workflow

```
1. USER CAPTURES / UPLOADS PHOTO
   │
   ├─► Camera Feed (Webcam / USB) OR File Upload OR Dataset Picker
   │
2. BROWSER PACKAGES MULTIMODAL PAYLOAD
   │
   ├─► formData.append('image', blob)
   ├─► formData.append('mq135', live_mq135)
   ├─► formData.append('mq2', live_mq2)
   │
3. POST TO /api/analyze (backend/app.py)
   │
   ├─► Step A: cv_analyzer.py processes image:
   │   - Segments green/charcoal pixels in HSV
   │   - Computes mold_coverage_pct & texture_roughness
   │   - Generates base64 annotated diagnostic image
   │
   ├─► Step B: model.py builds 16-feature vector
   │   - Combines live gas telemetry with CV metrics
   │   - Invokes RandomForestRegressor
   │   - Applies food safety gating rules
   │
   ├─► Step C: Archival:
   │   - Saves snapshot to scans/scan_YYYYMMDD_HHMMSS.jpg
   │   - Logs record to in-memory session history
   │
4. RESPONSE SENT TO FRONTEND (JSON)
   │
   ├─► Updates Shelf Life Headline ("53.5 Hours Remaining")
   ├─► Updates Freshness Donut Chart & Tooltip
   ├─► Updates Bubble Rating Badges (95% Fresh, 92% Edible)
   ├─► Renders CV Annotated Image with Bounding Boxes
   └─► Appends new record to History Table
```

---

## 10. Summary of Completed Work & Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **FastAPI Backend Server** | [`backend/app.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/backend/app.py) | REST API endpoints for `/api/analyze`, `/api/arduino/*`, `/api/cameras/*`, and `/api/dataset/*`. |
| **Arduino Serial Manager** | [`backend/arduino_manager.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/backend/arduino_manager.py) | Background serial listener for COM5, dynamic port scanner, and simulation fallback. |
| **Computer Vision Engine** | [`backend/cv_analyzer.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/backend/cv_analyzer.py) | HSV mold segmentation, Laplacian texture analysis, and diagnostic bounding-box annotation. |
| **Multimodal Predictor** | [`backend/model.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/backend/model.py) | 16-feature multimodal inference engine fusing IoT gas readings with vision features. |
| **Model Training Pipeline** | [`train_multimodal_model.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/train_multimodal_model.py) | Feature extractor and cross-validated Random Forest trainer ($R^2 = 0.9745$). |
| **Trained Model Artifact** | [`backend/shelf_life_multimodal.joblib`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/backend/shelf_life_multimodal.joblib) | Serialized multimodal model bundle with metadata and training metrics. |
| **Organic Bread Dataset** | [`bread_dataset/.../images`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/bread_dataset/bread_dataset/bread_dataset/synth_run1/images) | 50 photo-realistic bread lifecycle images based on user's authentic camera capture. |
| **Synchronized Telemetry** | [`sensor_data.csv`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/sensor_data.csv) | 70-hour time-series of MQ-135, MQ-2, temperature, and humidity. |
| **Figma Dashboard HTML** | [`static/index.html`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/static/index.html) | Figma Community light theme dashboard with sidebar, top bar, and 5 metric cards. |
| **Figma Dashboard Styles** | [`static/css/dashboard.css`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/static/css/dashboard.css) | Modern CSS tokens, double bar chart styles, floating tooltip, and circular bubble badges. |
| **Dashboard Controller** | [`static/js/dashboard.js`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/static/js/dashboard.js) | Frontend controller for WebRTC camera capture, Arduino polling, and prediction rendering. |
| **One-Click Startup Script** | [`run_dashboard.py`](file:///c:/Users/a/Documents/Daksh/Sem3/Embedded%20Systems/project/run_dashboard.py) | Port checks, browser launch, and Uvicorn server startup. |

---

### How to Run:
```powershell
python run_dashboard.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any browser.
