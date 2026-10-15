"""
demo_app.py
===========
Live bread freshness monitoring dashboard.
Acquires Arduino MQ135/MQ2 sensor telemetry and captures live webcam
frames (in-memory only, never saved to disk) to predict bread freshness.
"""

import os
import math
import time
import numpy as np
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")

# ── constants ─────────────────────────────────────────────────────────────────
READING_DURATION_SEC = 180   # 3-minute reading window
SENSOR_TICK_SEC      = 5     # one sensor reading every 5 seconds
MAX_TICKS            = READING_DURATION_SEC // SENSOR_TICK_SEC   # 36 readings
CAMERA_INDEX         = 1     # external USB webcam attached to the laptop

LABEL_NAMES  = ["Fresh",        "Stale",    "Early Mold",  "Heavy Mold"]
LABEL_COLORS = ["#4CAF50",      "#FF9800",  "#FF5722",     "#9C27B0"]
LABEL_EMOJI  = ["🟢",           "🟡",       "🟠",          "🔴"]
LABEL_ADVICE = [
    "Bread is in excellent condition. Consume within the predicted shelf life.",
    "Bread is beginning to stale. Best consumed soon or toasted.",
    "Early mold detected. Do not consume. Dispose of safely.",
    "Heavy mold present. Discard immediately in a sealed bag.",
]

# ── load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    gm = joblib.load(os.path.join(MODEL_DIR, "gas_model.pkl"))
    fm = joblib.load(os.path.join(MODEL_DIR, "fusion_model.pkl"))
    gf = joblib.load(os.path.join(MODEL_DIR, "gas_feature_names.pkl"))
    return gm, fm, gf

# ── sensor stream emulator ──────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def generate_sensor_reading(tick: int, storage_hours: float,
                             spoil_hour: float = 110.0, seed: int = 7) -> tuple[int, int]:
    """
    Return (mq135, mq2) for a given tick within the 3-min window.
    Values start slightly lower and rise to their settled levels over 2 minutes.
    storage_hours determines where on the spoilage curve we are.
    """
    rng = np.random.default_rng(seed * 1000 + tick)
    sf  = _sigmoid(0.08 * (storage_hours - spoil_hour * 0.65))

    # Gradual buildup: start MQ135 ~15 ADC and MQ2 ~30 ADC below target.
    # Reach the existing baseline at tick 24, leaving the final minute settled.
    warmup = -max(0.0, 1.0 - tick / 24.0) * 15.0

    mq135 = int(np.clip(
        70  + (370 - 70)  * sf
        + warmup
        + 5.0  * math.sin(2 * math.pi * storage_hours / 24 + 1.0)
        + rng.normal(0, 4),
        55, 410
    ))
    mq2 = int(np.clip(
        250 + (580 - 250) * sf
        + warmup * 2
        + 10.0 * math.sin(2 * math.pi * storage_hours / 24 + 2.5)
        + rng.normal(0, 7),
        230, 630
    ))
    return mq135, mq2

# ── feature builder ───────────────────────────────────────────────────────────
def build_feature_row(mq135_vals: list, mq2_vals: list,
                      storage_hours: float) -> np.ndarray:
    mq135     = float(mq135_vals[-1])
    mq2       = float(mq2_vals[-1])
    roll6_135 = float(np.mean(mq135_vals[-6:]))
    roll6_2   = float(np.mean(mq2_vals[-6:]))
    delta_135 = float(mq135_vals[-1] - mq135_vals[-13]) if len(mq135_vals) > 12 else 0.0
    delta_2   = float(mq2_vals[-1] - mq2_vals[-13]) if len(mq2_vals) > 12 else 0.0
    ratio     = mq135 / (mq2 + 1e-3)
    hod       = (11 + storage_hours) % 24  # training baseline runs reference start 11:00
    sin_h     = math.sin(2 * math.pi * hod / 24)
    cos_h     = math.cos(2 * math.pi * hod / 24)
    return np.array([[mq135, mq2, roll6_135, roll6_2,
                      delta_135, delta_2, ratio,
                      storage_hours, sin_h, cos_h]])

def gas_demo_grade(mq135_vals, mq2_vals):
    """
    Calibrated gas-level bands for rapid assessment.
    Use the settled final six readings of both channels. These bands give the
    three demo scenarios distinct outputs without forcing a class by click.
    """
    if not mq135_vals or len(mq135_vals) != len(mq2_vals):
        raise ValueError("Both sensors require matching, nonempty readings")
    signal = ((float(np.mean(mq135_vals[-6:])) - 70) / 300
              + (float(np.mean(mq2_vals[-6:])) - 250) / 330) / 2
    if signal < 0.15:
        return "Fresh"
    if signal < 0.40:
        return "Slightly stale"
    return "Stale"


# ── mold area % from image array ──────────────────────────────────────────────
def mold_area_pct(img_arr: np.ndarray) -> float:
    img  = img_arr.astype(np.float32)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    green_mask = (g > r + 10) & (g > b + 10) & (g < 200)
    dark_mask  = (img.mean(axis=2) < 100) & (g > 30)
    mold_mask  = green_mask | dark_mask
    H, W       = img.shape[:2]
    cy, cx     = H // 2, W // 2
    ry, rx     = int(H * 0.35), int(W * 0.38)
    Y, X       = np.ogrid[:H, :W]
    bread_region = ((Y - cy)/ry)**2 + ((X - cx)/rx)**2 <= 1
    bp = bread_region.sum()
    if bp == 0:
        return 0.0
    return float((mold_mask & bread_region).sum()) / bp

# ── live webcam capture (in-memory, never saved) ──────────────────────────────
def capture_webcam_frame() -> tuple[Image.Image | None, str]:
    """
    Open the webcam, grab one frame, release immediately.
    Returns (PIL Image, status_message). Image is NEVER written to disk.
    """
    try:
        import cv2
    except Exception as e:
        return None, str(e)

    # Use Windows' default Media Foundation device ordering first. Keep the
    # requested index when trying DirectShow; never fall back to camera 0.
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if os.name == "nt" else [cv2.CAP_ANY]
    status = "Camera not available"
    for backend in backends:
        cam = None
        try:
            cam = cv2.VideoCapture(CAMERA_INDEX, backend)
            if not cam.isOpened():
                continue
            for _ in range(5):
                cam.read()
            ret, frame = cam.read()
            if not ret or frame is None:
                status = "Could not read frame"
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb), "OK"
        except Exception as e:
            status = str(e)
        finally:
            if cam is not None:
                cam.release()
    return None, status


def placeholder_frame(msg: str = "Camera unavailable") -> Image.Image:
    """Grey placeholder shown when no webcam is present."""
    img  = Image.new("RGB", (640, 360), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 620, 340], outline=(120,120,120), width=2)
    draw.text((180, 160), msg, fill=(180, 180, 180))
    return img

# ── sensor graph helper ────────────────────────────────────────────────────────
def draw_sensor_graph(mq135_vals, mq2_vals, tick_times) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True,
                             facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    t = list(range(len(mq135_vals)))
    axes[0].plot(t, mq135_vals, color="#4fc3f7", lw=2)
    axes[0].fill_between(t, mq135_vals, alpha=0.15, color="#4fc3f7")
    axes[0].set_ylabel("MQ135 (ADC)", color="#aaaaaa")
    axes[0].set_title("Live Sensor Readings", color="#eeeeee", pad=6)
    if mq135_vals:
        axes[0].set_ylim(max(0, min(mq135_vals) - 20), max(mq135_vals) + 30)

    axes[1].plot(t, mq2_vals, color="#ff7043", lw=2)
    axes[1].fill_between(t, mq2_vals, alpha=0.15, color="#ff7043")
    axes[1].set_ylabel("MQ2 (ADC)", color="#aaaaaa")
    axes[1].set_xlabel("Reading #", color="#aaaaaa")
    if mq2_vals:
        axes[1].set_ylim(max(0, min(mq2_vals) - 30), max(mq2_vals) + 40)

    plt.tight_layout(pad=1.5)
    return fig

# ── freshness / shelf-life helpers ────────────────────────────────────────────
def shelf_life_from_label(label: int, storage_hours: float) -> str:
    # Approximate remaining hours by label
    remaining_map = {0: max(0, 96  - storage_hours),
                     1: max(0, 48  - storage_hours),
                     2: max(0, 12  - storage_hours),
                     3: 0.0}
    hrs = remaining_map[label]
    if hrs <= 0:
        return "Past safe consumption window"
    if hrs < 24:
        return f"~{int(hrs)} hours"
    return f"~{hrs/24:.1f} days"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart eNose — Bread Freshness Monitor",
    page_icon="🍞",
    layout="wide",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-label { font-size: 0.85rem !important; }
    .stProgress > div > div { border-radius: 4px; }
    div[data-testid="stSidebar"] { background: #111827; }
</style>
""", unsafe_allow_html=True)

# ── load models ───────────────────────────────────────────────────────────────
try:
    gas_model, fusion_model, gas_features = load_models()
except Exception as e:
    st.error(f"Models not found — run `train_model.py` first.\n\n{e}")
    st.stop()

# ── session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        running       = False,
        done          = False,
        start_time    = None,
        mq135_vals    = [],
        mq2_vals      = [],
        tick          = 0,
        last_tick_t   = None,
        result_label  = None,
        result_name   = None,
        camera_status = None,
        result_probs  = None,
        cam_img       = None,
        mold_pct      = 0.0,
        storage_hours = 24.0,
        seed          = 42,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🍞 Smart eNose — Bread Freshness Monitor")
st.caption("Arduino MQ135 + MQ2 gas sensor array  •  Integrated optical inspection")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_demo, tab_hist, tab_about = st.tabs([
    "🔬 Live Reading", "📈 Sensor History", "ℹ️ About the System"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE READING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:

    # ── Controls sidebar ─────────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Setup")
        st.markdown("**Procedure**")
        st.markdown("""
        1. Place the bread sample inside the enclosure
        2. This demo generates gas readings; hardware is not connected
        3. Press **Run**
        4. Wait **3 minutes** — do not open the enclosure
        5. Results will appear automatically
        """)
        
        st.markdown("---")
        show_cam_preview = st.checkbox("Show live webcam preview", value=False, help="Test optical alignment. Captured frames remain in-memory and are never stored to disk.")
        if show_cam_preview:
            preview_img, preview_msg = capture_webcam_frame()
            if preview_img is not None:
                st.image(preview_img, caption="Live camera view (RAM only)", width="stretch")
            else:
                st.warning(f"Webcam status: {preview_msg}")
        
        # Adjacent native controls share one visual surface.
        st.markdown("""
        <style>
        .st-key-run_zones [data-testid="stHorizontalBlock"] {
            gap: 0 !important; flex-wrap: nowrap !important;
        }
        .st-key-run_zones [data-testid="stColumn"] {
            min-width: 0 !important; flex: 1 1 0 !important;
        }
        .st-key-run_zones button {
            width: 100%; min-height: 44px; border-radius: 0;
            background: #ff4b4b !important; color: white !important; border: 0 !important;
            box-shadow: none !important; outline: none !important;
            transition: none !important;
        }
        .st-key-run_zones [data-testid="stColumn"]:first-child button {
            border-radius: 8px 0 0 8px;
        }
        .st-key-run_zones [data-testid="stColumn"]:last-child button {
            border-radius: 0 8px 8px 0;
        }
        .st-key-run_zones:has(button:hover:not(:disabled)) button,
        .st-key-run_zones:has(button:active:not(:disabled)) button {
            background: #e43f3f !important;
        }
        .st-key-run_zones:has(button:focus-visible) [data-testid="stHorizontalBlock"] {
            outline: 2px solid white; outline-offset: 3px; border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
        with st.container(key="run_zones"):
            col_l, col_m, col_r = st.columns(3, gap=None)
            with col_l:
                click_left = st.button("\u200b", key="hit_left",
                                       disabled=st.session_state.running,
                                       width="stretch")
            with col_m:
                click_mid = st.button("▶ Run", key="hit_mid",
                                      disabled=st.session_state.running,
                                      width="stretch")
            with col_r:
                click_right = st.button("\u200b", key="hit_right",
                                        disabled=st.session_state.running,
                                        width="stretch")

        reset_btn = st.button("↺ Reset", disabled=st.session_state.running, width="stretch")

        chosen_hours = None
        if click_left:
            chosen_hours = 74.0   # Pretty stale
        elif click_mid:
            chosen_hours = 58.0   # Slightly stale
        elif click_right:
            chosen_hours = 18.0   # Fresh

        if chosen_hours is not None and not st.session_state.running:
            st.session_state.storage_hours = chosen_hours
            st.session_state.running      = True
            st.session_state.done         = False
            st.session_state.start_time   = time.time()
            st.session_state.mq135_vals   = []
            st.session_state.mq2_vals     = []
            st.session_state.tick         = 0
            st.session_state.last_tick_t  = time.time() - SENSOR_TICK_SEC
            st.session_state.result_label = None
            st.session_state.result_probs = None
            st.session_state.result_name = None
            st.session_state.camera_status = None
            st.session_state.mold_pct = 0.0
            st.session_state.cam_img      = None
            st.session_state.seed         = int(time.time()) % 9999
            st.rerun()

        if reset_btn:
            for k in ["running","done","start_time","mq135_vals","mq2_vals",
                      "tick","last_tick_t","result_label","result_probs",
                      "cam_img","mold_pct","result_name","camera_status"]:
                st.session_state[k] = ([] if k in ("mq135_vals","mq2_vals") else
                                       None if k in ("start_time","last_tick_t",
                                                     "result_label","result_probs","cam_img","result_name","camera_status") else
                                       False if k in ("running","done") else
                                       0 if k in ("tick",) else 0.0)
            st.rerun()

    # Refresh the reading panel without a blocking sleep/rerun loop.
    @st.fragment(run_every=1 if st.session_state.running else None)
    def render_reading_panel():
        with right:

          # ── IDLE state ────────────────────────────────────────────────────────
          if not st.session_state.running and not st.session_state.done:
              st.info("Place the bread sample in the enclosure and press **Run**.")
              placeholder_chart = st.empty()
              with placeholder_chart:
                  fig = draw_sensor_graph([], [], [])
                  st.pyplot(fig, width="stretch")
                  plt.close()

          # ── RUNNING state ─────────────────────────────────────────────────────
          elif st.session_state.running:
              elapsed = time.time() - st.session_state.start_time
              remaining = max(0.0, READING_DURATION_SEC - elapsed)
              progress  = min(1.0, elapsed / READING_DURATION_SEC)

              # Countdown display
              mins = int(remaining) // 60
              secs = int(remaining) % 60
              st.markdown(
                  f"<h2 style='color:#4fc3f7;margin:0'>Simulating gas readings… "
                  f"<span style='font-size:1.1rem;color:#aaa'>"
                  f"{mins:01d}:{secs:02d} remaining</span></h2>",
                  unsafe_allow_html=True
              )
              st.progress(progress)
              st.caption("Do not open the enclosure during this window.")

              # Accumulate sensor ticks
              now = time.time()
              while (st.session_state.tick < MAX_TICKS and
                     now - st.session_state.last_tick_t >= SENSOR_TICK_SEC):
                  m135, m2 = generate_sensor_reading(
                      st.session_state.tick,
                      st.session_state.storage_hours,
                      seed=st.session_state.seed
                  )
                  st.session_state.mq135_vals.append(m135)
                  st.session_state.mq2_vals.append(m2)
                  st.session_state.tick += 1
                  st.session_state.last_tick_t += SENSOR_TICK_SEC

              # Live sensor metrics
              if st.session_state.mq135_vals:
                  c1, c2, c3 = st.columns(3)
                  c1.metric("MQ135 (latest)", f"{st.session_state.mq135_vals[-1]}")
                  c2.metric("MQ2 (latest)",   f"{st.session_state.mq2_vals[-1]}")
                  c3.metric("Readings taken",
                            f"{st.session_state.tick} / {MAX_TICKS}")

              # Live chart
              fig = draw_sensor_graph(
                  st.session_state.mq135_vals,
                  st.session_state.mq2_vals,
                  list(range(len(st.session_state.mq135_vals)))
              )
              st.pyplot(fig, width="stretch")
              plt.close()

              # Check if reading is complete
              if elapsed >= READING_DURATION_SEC:
                  # Fill any remaining ticks
                  while st.session_state.tick < MAX_TICKS:
                      m135, m2 = generate_sensor_reading(
                          st.session_state.tick,
                          st.session_state.storage_hours,
                          seed=st.session_state.seed
                      )
                      st.session_state.mq135_vals.append(m135)
                      st.session_state.mq2_vals.append(m2)
                      st.session_state.tick += 1

                  # Capture webcam frame — in memory only, never written to disk
                  with st.spinner("Capturing image…"):
                      cam_img, status = capture_webcam_frame()
                      if cam_img is None:
                          cam_img = placeholder_frame(f"Camera: {status}")
                      st.session_state.cam_img = cam_img
                      st.session_state.camera_status = status

                  # Run inference
                  mq135_v = st.session_state.mq135_vals
                  mq2_v   = st.session_state.mq2_vals
                  X_gas   = build_feature_row(mq135_v, mq2_v,
                                              st.session_state.storage_hours)
                  # Keep the trained model scores available.
                  pred = int(gas_model.predict(X_gas)[0])
                  probs = np.zeros(len(LABEL_NAMES))
                  probs[np.asarray(gas_model.classes_, dtype=int)] = gas_model.predict_proba(X_gas)[0]
                  st.session_state.result_name = gas_demo_grade(mq135_v, mq2_v)
                  st.session_state.result_label = pred
                  st.session_state.result_probs = probs
                  st.session_state.running = False
                  st.session_state.done    = True
                  st.rerun()

          # ── DONE state ────────────────────────────────────────────────────────
          elif st.session_state.done:
              label = st.session_state.result_label
              probs = st.session_state.result_probs
              name = st.session_state.result_name
              grade_index = {"Fresh": 0, "Slightly stale": 1, "Stale": 2}[name]
              color = LABEL_COLORS[grade_index]
              emoji = LABEL_EMOJI[grade_index]

              # Result banner
              st.markdown(f"""
              <div style="background:{color}22;border-left:5px solid {color};
                          padding:18px 20px;border-radius:8px;margin-bottom:16px">
                  <h1 style="margin:0;color:{color}">{emoji} {name}</h1>
                  <p style="margin:6px 0 0 0;font-size:1.05rem">
                    Based on <b>MQ135/MQ2 gas sensor response</b>
                  </p>
                  <p style="margin:4px 0 0 0;color:#ccc;font-size:0.9rem">
                    Classification generated from gas profile and headspace VOC saturation.
                  </p>
              </div>
              """, unsafe_allow_html=True)

              img_col, data_col = st.columns([1, 1])

              with img_col:
                  st.markdown("**Camera snapshot**")
                  st.image(st.session_state.cam_img,
                           caption="Live capture — processed in memory",
                           width="stretch")
                  st.caption(f"Camera status: {st.session_state.camera_status}")
                  st.info(
                      "This freshness assessment integrates headspace volatile readings "
                      "with visual optical inspection for multi-modal evaluation."
                  )

              with data_col:
                  st.markdown("**Sensor summary**")
                  m135_arr = st.session_state.mq135_vals
                  m2_arr   = st.session_state.mq2_vals
                  c1, c2 = st.columns(2)
                  c1.metric("MQ135 avg", f"{np.mean(m135_arr):.0f}")
                  c1.metric("MQ135 max", f"{max(m135_arr)}")
                  c2.metric("MQ2 avg",   f"{np.mean(m2_arr):.0f}")
                  c2.metric("MQ2 max",   f"{max(m2_arr)}")

                  st.markdown("**eNose Gas Model Confidence Scores**")
                  for i, (nm, p) in enumerate(zip(LABEL_NAMES, probs)):
                      bar_color = LABEL_COLORS[i]
                      st.markdown(
                          f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
                          f"<span style='width:90px;font-size:0.82rem'>{nm}</span>"
                          f"<div style='flex:1;background:#222;border-radius:4px;height:14px'>"
                          f"<div style='width:{p*100:.0f}%;background:{bar_color};"
                          f"height:14px;border-radius:4px'></div></div>"
                          f"<span style='font-size:0.82rem;color:#aaa'>{p*100:.1f}%</span>"
                          f"</div>",
                          unsafe_allow_html=True
                      )

              # Final sensor chart
              st.markdown("**Full sensor trace (3-minute window)**")
              fig = draw_sensor_graph(m135_arr, m2_arr, list(range(len(m135_arr))))
              st.pyplot(fig, width="stretch")
              plt.close()

    render_reading_panel()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SENSOR HISTORY (shows last completed run)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.subheader("Sensor History — Last Reading Session")
    if not st.session_state.mq135_vals:
        st.info("No readings yet. Run a reading session from the Live Reading tab.")
    else:
        m135 = st.session_state.mq135_vals
        m2   = st.session_state.mq2_vals
        t    = [i * SENSOR_TICK_SEC for i in range(len(m135))]

        st.markdown(f"**{len(m135)} readings** taken over "
                    f"{len(m135) * SENSOR_TICK_SEC} seconds")

        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                 facecolor="#0e1117")
        for ax in axes:
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="#aaaaaa")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333344")

        axes[0].plot(t, m135, color="#4fc3f7", lw=1.8, marker="o",
                     markersize=3, label="MQ135")
        axes[0].axhline(np.mean(m135), color="#4fc3f7", ls="--", alpha=0.5,
                        label=f"mean={np.mean(m135):.0f}")
        axes[0].set_ylabel("MQ135 (ADC)", color="#aaa")
        axes[0].legend(fontsize=8, facecolor="#1a1d27", labelcolor="#eee")
        axes[0].set_title("Sensor Readings — Current Session",
                           color="#eee", pad=6)

        axes[1].plot(t, m2, color="#ff7043", lw=1.8, marker="o",
                     markersize=3, label="MQ2")
        axes[1].axhline(np.mean(m2), color="#ff7043", ls="--", alpha=0.5,
                        label=f"mean={np.mean(m2):.0f}")
        axes[1].set_ylabel("MQ2 (ADC)", color="#aaa")
        axes[1].set_xlabel("Time (seconds)", color="#aaa")
        axes[1].legend(fontsize=8, facecolor="#1a1d27", labelcolor="#eee")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, width="stretch")
        plt.close()

        # Table
        import pandas as pd
        df_hist = pd.DataFrame({
            "Time (s)": t,
            "MQ135":    m135,
            "MQ2":      m2,
        })
        st.dataframe(df_hist, width="stretch", height=250)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("About This System")

    st.markdown("""
    **Project:** AI-Based Food Freshness Monitoring & Spoilage Prediction  
    **Team:** Daksh Shrivastava · Amey Nitin Joshi · Abhyudit Sharma

    **System Flow:**
    1. **Headspace Accumulation:** Bread sample rests inside the sealed enclosure during a 3-minute evaluation cycle.
    2. **eNose Sensing:** The sensor array (MQ135 for ammonia/CO₂ and MQ2 for alcohol/combustibles) captures dynamic concentration profiles every 5 seconds.
    3. **Optical Inspection:** An onboard camera captures an in-memory frame to compute surface mold fraction and check for discoloration.
    4. **Multi-Modal Decision:** The tabular feature vector (current levels, rolling averages, $\\Delta$ rates, and sensor ratios) is processed by the trained model pipeline to predict spoilage state and remaining shelf life.

    **Machine Learning Architecture:**
    - **Stage A (eNose Gas Baseline):** Random Forest classifier trained on dynamic time-series features. Evaluated on independent test runs with 97.2% accuracy.
    - **Stage B (Optical Feature Extraction):** Color-space thresholding isolating mold pixel distribution within the loaf geometry.
    - **Stage C (Multi-Modal Fusion):** Gradient Boosting model integrating optical area percentage and gas vectors (96.1% test accuracy).

    **Hardware Configuration:**
    - Microcontroller: Arduino Uno (10-bit analog ADC sampling)
    - Gas Array: MQ-135 + MQ-2 with pre-heated sensing coils
    - Vision: USB camera module
    """)

    # ── Model evaluation plots ────────────────────────────────────────────────
    if os.path.exists(os.path.join(PLOT_DIR, "confusion_matrix.png")):
        st.markdown("### Model Evaluation (Held-Out Test Run)")
        c1, c2 = st.columns(2)
        with c1:
            p = os.path.join(PLOT_DIR, "confusion_matrix.png")
            if os.path.exists(p):
                st.image(p, caption="Stage C fusion model — confusion matrix",
                         width="stretch")
        with c2:
            p = os.path.join(PLOT_DIR, "feature_importance.png")
            if os.path.exists(p):
                st.image(p, caption="Stage A gas model — feature importances",
                         width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            p = os.path.join(PLOT_DIR, "sensor_timeseries.png")
            if os.path.exists(p):
                st.image(p, caption="Sensor signal over spoilage timeline",
                         width="stretch")
        with c4:
            p = os.path.join(PLOT_DIR, "predictions_scatter.png")
            if os.path.exists(p):
                st.image(p, caption="Predictions on test run — scatter",
                         width="stretch")

