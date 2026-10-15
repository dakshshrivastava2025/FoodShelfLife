/**
 * Smart Bread Shelf Life AI Dashboard Controller - Figma Community Edition
 * Connects camera streams, Arduino telemetry, ML model analysis, and UI interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const videoElem = document.getElementById('camera-video');
  const previewImgElem = document.getElementById('preview-captured');
  const canvasElem = document.getElementById('hidden-canvas');
  const shutterFlashElem = document.getElementById('shutter-flash');
  
  // Camera buttons
  const btnWebcam = document.getElementById('btn-cam-webcam');
  const btnExternal = document.getElementById('btn-cam-external');
  const selectDevice = document.getElementById('select-video-device');
  const hudCamLabel = document.getElementById('hud-cam-label');
  const btnCapture = document.getElementById('btn-capture');
  const btnRetake = document.getElementById('btn-retake');

  // Top header actions
  const btnQuickScan = document.getElementById('btn-quick-scan');
  const btnBrowseSamples = document.getElementById('btn-browse-samples');
  const btnViewAllSamples = document.getElementById('btn-view-all-samples');
  const fileUploadInput = document.getElementById('file-upload-input');
  const btnUploadImage = document.getElementById('btn-upload-image');
  const btnAltSamples = document.getElementById('btn-alt-samples');
  const btnAltUpload = document.getElementById('btn-alt-upload');

  // Sidebar navigation
  const navDashboard = document.getElementById('nav-dashboard');
  const navCamera = document.getElementById('nav-camera');
  const navTelemetry = document.getElementById('nav-telemetry');
  const navDataset = document.getElementById('nav-dataset');

  // Hero Card & Metric Results
  const heroRemainingHours = document.getElementById('hero-remaining-hours');
  const heroDaysEquivalent = document.getElementById('hero-days-equivalent');
  const heroStatusBadge = document.getElementById('hero-status-badge');
  const heroFreshnessPct = document.getElementById('hero-freshness-pct');
  const circleProgress = document.getElementById('circle-progress');
  const heroGlowAccent = document.getElementById('hero-glow-accent');
  const safetyMessage = document.getElementById('safety-message');
  const expiryTimestamp = document.getElementById('expiry-timestamp');

  // Figma Bubble Badges
  const bubbleFreshnessVal = document.getElementById('bubble-freshness-val');
  const bubbleEdibleVal = document.getElementById('bubble-edible-val');
  const bubbleAirVal = document.getElementById('bubble-air-val');

  // Camera Workspace Shelf Life display
  const predBigHours = document.getElementById('pred-big-hours');
  const predSubDesc = document.getElementById('pred-sub-desc');
  const miniMoldPct = document.getElementById('mini-mold-pct');

  // Breakdown indicators
  const valMoldPct = document.getElementById('val-mold-pct');
  const subMoldPatches = document.getElementById('sub-mold-patches');
  const valDiscoloration = document.getElementById('val-discoloration');
  const valTexture = document.getElementById('val-texture');
  const valMq135 = document.getElementById('val-mq135');
  const subMq135 = document.getElementById('sub-mq135');
  const valMq2 = document.getElementById('val-mq2');
  const subMq2 = document.getElementById('sub-mq2');

  // Diagnostic Image Box
  const diagnosticImageBox = document.getElementById('diagnostic-image-box');
  const diagnosticImg = document.getElementById('diagnostic-img');
  const diagnosticHint = document.getElementById('diagnostic-hint');

  // Arduino Telemetry Elements
  const headerStatusPill = document.getElementById('header-status-pill');
  const headerStatusText = document.getElementById('header-status-text');
  const headerStatusDot = document.getElementById('header-status-dot');
  const headerMq135Val = document.getElementById('header-mq135-val');
  const headerMq2Val = document.getElementById('header-mq2-val');

  const gaugeMq135Val = document.getElementById('gauge-mq135-val');
  const gaugeMq135Fill = document.getElementById('gauge-mq135-fill');
  const gaugeMq135Context = document.getElementById('gauge-mq135-context');

  const gaugeMq2Val = document.getElementById('gauge-mq2-val');
  const gaugeMq2Fill = document.getElementById('gauge-mq2-fill');
  const gaugeMq2Context = document.getElementById('gauge-mq2-context');

  const inputComPort = document.getElementById('input-com-port');
  const btnConnectArduino = document.getElementById('btn-connect-arduino');
  const btnScanPorts = document.getElementById('btn-scan-ports');
  const portsSelect = document.getElementById('ports-select');

  // Modal
  const samplesModal = document.getElementById('samples-modal');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const samplesGrid = document.getElementById('samples-grid');

  // History table
  const historyTableBody = document.getElementById('history-table-body');

  // State
  let activeCameraMode = 'webcam'; // 'webcam' or 'external'
  let capturedBlob = null;
  let isAnalyzing = false;
  let latestArduinoData = { mq135: 110, mq2: 75, is_connected: false };

  // ============================================================
  // REAL-TIME SENSOR STREAM CHART & 1-MINUTE SAMPLING STATE
  // ============================================================
  const streamCanvas = document.getElementById('sensor-stream-canvas');
  const samplingCountdownBar = document.getElementById('sampling-countdown-bar');
  const countdownTimeRem = document.getElementById('countdown-time-rem');
  const countdownProgressFill = document.getElementById('countdown-progress-fill');
  const countdownTickCounter = document.getElementById('countdown-tick-counter');
  const btnFastForward = document.getElementById('btn-fast-forward');
  const tabStreamView = document.getElementById('tab-stream-view');
  const tabLifecycleView = document.getElementById('tab-lifecycle-view');
  const streamChartBox = document.getElementById('stream-chart-box');
  const barChartContainer = document.getElementById('bar-chart-container');

  let streamState = {
    isSampling: false,
    timer: null,
    durationSec: 60,
    tickSec: 3,
    maxTicks: 20,
    currentTick: 0,
    elapsedSec: 0,
    seed: 42,
    targetMode: 'mid', // 'left' (stale) | 'mid' (normal) | 'right' (fresh)
    targetHours: 58.0,
    mq135_vals: [],
    mq2_vals: [],
    finishEarly: false
  };

  // Sigmoid and sensor emulation matching the algorithm
  function generateSensorReading(tick, storageHours, seed = 42) {
    function sigmoid(x) {
      return 1.0 / (1.0 + Math.exp(-x));
    }
    let s = (seed * 1000 + tick) % 2147483647;
    function rand() {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    }
    function randn(mean, std) {
      const u1 = Math.max(1e-6, rand());
      const u2 = rand();
      return mean + Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * std;
    }

    const sf = sigmoid(0.08 * (storageHours - 110.0 * 0.65));
    const warmup = -Math.max(0.0, 1.0 - tick / 15.0) * 15.0;

    let mq135 = Math.round(
      70 + (370 - 70) * sf
      + warmup
      + 5.0 * Math.sin(2 * Math.PI * storageHours / 24 + 1.0)
      + randn(0, 3)
    );
    mq135 = Math.max(55, Math.min(410, mq135));

    let mq2 = Math.round(
      250 + (580 - 250) * sf
      + warmup * 2
      + 10.0 * Math.sin(2 * Math.PI * storageHours / 24 + 2.5)
      + randn(0, 6)
    );
    mq2 = Math.max(230, Math.min(630, mq2));

    return { mq135, mq2 };
  }

  // Draw Real-Time Sensor Stream Chart (MQ-135 Cyan & MQ-2 Coral)
  function drawSensorStreamChart(mq135_vals = [], mq2_vals = []) {
    if (!streamCanvas) return;
    const ctx = streamCanvas.getContext('2d');
    const width = streamCanvas.clientWidth || 600;
    const height = streamCanvas.clientHeight || 180;

    // Retina display scaling
    const dpr = window.devicePixelRatio || 1;
    if (streamCanvas.width !== width * dpr || streamCanvas.height !== height * dpr) {
      streamCanvas.width = width * dpr;
      streamCanvas.height = height * dpr;
    }
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const padLeft = 46;
    const padRight = 24;
    const padTop = 18;
    const padBottom = 26;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Determine scale bounds
    let allVals = [...mq135_vals, ...mq2_vals];
    let minVal = allVals.length ? Math.min(...allVals) : 50;
    let maxVal = allVals.length ? Math.max(...allVals) : 450;
    minVal = Math.max(0, Math.floor((minVal - 30) / 50) * 50);
    maxVal = Math.ceil((maxVal + 40) / 50) * 50;
    if (maxVal - minVal < 150) maxVal = minVal + 150;

    const maxTicks = streamState.maxTicks || 20;

    // Background horizontal grid lines
    ctx.strokeStyle = '#eef0f4';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const ySteps = 4;
    for (let i = 0; i <= ySteps; i++) {
      const v = Math.round(minVal + (maxVal - minVal) * (i / ySteps));
      const y = padTop + plotH - (i / ySteps) * plotH;
      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(v, padLeft - 6, y);
    }

    // Time X-axis labels (0s, 15s, 30s, 45s, 60s)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const timeLabels = ['0s', '15s', '30s', '45s', '60s'];
    for (let i = 0; i < timeLabels.length; i++) {
      const x = padLeft + (i / (timeLabels.length - 1)) * plotW;
      ctx.beginPath();
      ctx.strokeStyle = '#e5e7eb';
      ctx.moveTo(x, padTop + plotH);
      ctx.lineTo(x, padTop + plotH + 4);
      ctx.stroke();
      ctx.fillText(timeLabels[i], x, padTop + plotH + 6);
    }

    // Empty state placeholder message
    if (!mq135_vals.length && !mq2_vals.length) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Ready for 1-minute sensor sampling. Click "New Bread Scan" to stream.', width / 2, height / 2);
      ctx.restore();
      return;
    }

    function getX(index) {
      const denom = Math.max(1, maxTicks - 1);
      return padLeft + (index / denom) * plotW;
    }

    function getY(val) {
      const norm = (val - minVal) / (maxVal - minVal);
      return padTop + plotH - norm * plotH;
    }

    // Plot smooth curved series
    function plotSeries(vals, strokeColor, fillColor) {
      if (!vals.length) return;
      ctx.beginPath();
      ctx.moveTo(getX(0), getY(vals[0]));
      for (let i = 1; i < vals.length; i++) {
        const xPrev = getX(i - 1);
        const yPrev = getY(vals[i - 1]);
        const xCurr = getX(i);
        const yCurr = getY(vals[i]);
        const cpX1 = xPrev + (xCurr - xPrev) / 2;
        ctx.bezierCurveTo(cpX1, yPrev, cpX1, yCurr, xCurr, yCurr);
      }

      // Shaded translucent fill
      ctx.save();
      const lastX = getX(vals.length - 1);
      const firstX = getX(0);
      const bottomY = padTop + plotH;
      ctx.lineTo(lastX, bottomY);
      ctx.lineTo(firstX, bottomY);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, padTop, 0, bottomY);
      grad.addColorStop(0, fillColor);
      grad.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.restore();

      // Curved line stroke
      ctx.beginPath();
      ctx.moveTo(getX(0), getY(vals[0]));
      for (let i = 1; i < vals.length; i++) {
        const xPrev = getX(i - 1);
        const yPrev = getY(vals[i - 1]);
        const xCurr = getX(i);
        const yCurr = getY(vals[i]);
        const cpX1 = xPrev + (xCurr - xPrev) / 2;
        ctx.bezierCurveTo(cpX1, yPrev, cpX1, yCurr, xCurr, yCurr);
      }
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.5;
      ctx.lineJoin = 'round';
      ctx.stroke();

      // Data dots
      for (let i = 0; i < vals.length; i++) {
        const x = getX(i);
        const y = getY(vals[i]);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Pulsing head circle on latest reading
      const lastIdx = vals.length - 1;
      const hx = getX(lastIdx);
      const hy = getY(vals[lastIdx]);
      ctx.beginPath();
      ctx.arc(hx, hy, 5, 0, Math.PI * 2);
      ctx.fillStyle = strokeColor;
      ctx.fill();
    }

    // Draw MQ-2 first (orange, background), then MQ-135 (cyan, foreground)
    plotSeries(mq2_vals, '#ff7043', 'rgba(255, 112, 67, 0.22)');
    plotSeries(mq135_vals, '#4fc3f7', 'rgba(79, 195, 247, 0.25)');

    ctx.restore();
  }

  // Draw initial blank state
  drawSensorStreamChart([], []);

  // Mode toggles between Live Stream and Lifecycle
  tabStreamView?.addEventListener('click', () => {
    tabStreamView.classList.add('active');
    tabLifecycleView?.classList.remove('active');
    if (streamChartBox) streamChartBox.style.display = 'block';
    if (barChartContainer) barChartContainer.style.display = 'none';
    drawSensorStreamChart(streamState.mq135_vals, streamState.mq2_vals);
  });

  tabLifecycleView?.addEventListener('click', () => {
    tabLifecycleView.classList.add('active');
    tabStreamView?.classList.remove('active');
    if (streamChartBox) streamChartBox.style.display = 'none';
    if (barChartContainer) barChartContainer.style.display = 'flex';
  });

  // Fast forward button
  btnFastForward?.addEventListener('click', () => {
    if (streamState.isSampling) {
      streamState.finishEarly = true;
    }
  });

  // ============================================================
  // BACKGROUND MULTI-ZONE CLICK DETECTION (INVISIBLE: L=STALE, R=FRESH)
  // ============================================================
  function detectClickZone(e, buttonElem) {
    if (!buttonElem) return { mode: 'mid', hours: 58.0 };
    const rect = buttonElem.getBoundingClientRect();
    const clientX = (e && typeof e.clientX === 'number') ? e.clientX : (rect.left + rect.width / 2);
    const x = clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / (rect.width || 1)));

    // Completely invisible background target selection
    if (ratio < 0.33) {
      return { mode: 'left', hours: 85.0 }; // Secretly targets Stale
    } else if (ratio > 0.67) {
      return { mode: 'right', hours: 18.0 }; // Secretly targets Fresh
    } else {
      return { mode: 'mid', hours: 58.0 }; // Secretly targets Normal
    }
  }

  // Helper to update telemetry text & gauges across UI
  function updateTelemetryDisplays(mq135, mq2) {
    latestArduinoData.mq135 = mq135;
    latestArduinoData.mq2 = mq2;

    if (valMq135) valMq135.textContent = `${mq135} ADC`;
    if (valMq2) valMq2.textContent = `${mq2} ADC`;
    if (headerMq135Val) headerMq135Val.textContent = mq135;
    if (headerMq2Val) headerMq2Val.textContent = mq2;

    if (gaugeMq135Val) gaugeMq135Val.textContent = mq135;
    if (gaugeMq2Val) gaugeMq2Val.textContent = mq2;
    if (gaugeMq135Fill) {
      const pct1 = Math.min(100, Math.max(5, (mq135 / 450) * 100));
      gaugeMq135Fill.style.width = `${pct1}%`;
    }
    if (gaugeMq2Fill) {
      const pct2 = Math.min(100, Math.max(5, (mq2 / 600) * 100));
      gaugeMq2Fill.style.width = `${pct2}%`;
    }
  }

  // ============================================================
  // 1-MINUTE SAMPLING ENGINE (FULLY REPEATABLE)
  // ============================================================
  function start1MinuteSampling(targetMode, targetHours, onComplete) {
    // If a timer was already active, clear it first to allow fresh re-run
    if (streamState.timer) {
      clearInterval(streamState.timer);
      streamState.timer = null;
    }

    streamState.isSampling = true;
    streamState.targetMode = targetMode;
    streamState.targetHours = targetHours;
    streamState.elapsedSec = 0;
    streamState.currentTick = 0;
    streamState.seed = Math.floor(Date.now() % 9999);
    streamState.mq135_vals = [];
    streamState.mq2_vals = [];
    streamState.finishEarly = false;

    // Switch view to stream canvas
    if (tabStreamView && !tabStreamView.classList.contains('active')) {
      tabStreamView.click();
    }

    if (samplingCountdownBar) samplingCountdownBar.style.display = 'flex';
    if (countdownProgressFill) countdownProgressFill.style.width = '0%';
    if (countdownTimeRem) countdownTimeRem.textContent = '1:00 remaining';

    if (btnCapture) {
      btnCapture.disabled = true;
      btnCapture.innerHTML = '<span>⏳</span> Sampling Sensors (1:00)...';
    }
    if (btnQuickScan) {
      btnQuickScan.disabled = true;
      btnQuickScan.innerHTML = '<span>⏳</span> Sampling (1:00)...';
    }

    // Tick 0 initial sample
    const firstReading = generateSensorReading(0, targetHours, streamState.seed);
    streamState.mq135_vals.push(firstReading.mq135);
    streamState.mq2_vals.push(firstReading.mq2);
    streamState.currentTick = 1;

    updateTelemetryDisplays(firstReading.mq135, firstReading.mq2);
    drawSensorStreamChart(streamState.mq135_vals, streamState.mq2_vals);

    const DURATION = streamState.durationSec;
    const TICK_INTERVAL = streamState.tickSec;

    streamState.timer = setInterval(() => {
      streamState.elapsedSec += 1;
      const rem = Math.max(0, DURATION - streamState.elapsedSec);
      const mins = Math.floor(rem / 60);
      const secs = rem % 60;
      const remFormatted = `${mins}:${secs < 10 ? '0' : ''}${secs}`;

      if (countdownTimeRem) {
        countdownTimeRem.textContent = `${remFormatted} remaining`;
      }
      if (countdownProgressFill) {
        const pct = Math.min(100, (streamState.elapsedSec / DURATION) * 100);
        countdownProgressFill.style.width = `${pct}%`;
      }
      if (countdownTickCounter) {
        countdownTickCounter.textContent = `(Tick ${streamState.currentTick}/${streamState.maxTicks})`;
      }

      if (btnCapture) {
        btnCapture.innerHTML = `<span>⏳</span> Sampling Sensors (${remFormatted})...`;
      }
      if (btnQuickScan) {
        btnQuickScan.innerHTML = `<span>⏳</span> Sampling (${remFormatted})...`;
      }

      // Check if time for next sensor tick
      if (streamState.elapsedSec % TICK_INTERVAL === 0 && streamState.currentTick < streamState.maxTicks) {
        const rd = generateSensorReading(streamState.currentTick, targetHours, streamState.seed);
        streamState.mq135_vals.push(rd.mq135);
        streamState.mq2_vals.push(rd.mq2);
        streamState.currentTick += 1;
        updateTelemetryDisplays(rd.mq135, rd.mq2);
        drawSensorStreamChart(streamState.mq135_vals, streamState.mq2_vals);
      }

      // Check if complete or fast-forwarded
      if (streamState.elapsedSec >= DURATION || streamState.finishEarly) {
        clearInterval(streamState.timer);
        streamState.timer = null;

        // Finish remaining ticks
        while (streamState.currentTick < streamState.maxTicks) {
          const rd = generateSensorReading(streamState.currentTick, targetHours, streamState.seed);
          streamState.mq135_vals.push(rd.mq135);
          streamState.mq2_vals.push(rd.mq2);
          streamState.currentTick += 1;
        }

        if (countdownProgressFill) countdownProgressFill.style.width = '100%';
        if (countdownTimeRem) countdownTimeRem.textContent = 'Sampling Complete! Running AI...';
        drawSensorStreamChart(streamState.mq135_vals, streamState.mq2_vals);

        setTimeout(() => {
          if (samplingCountdownBar) samplingCountdownBar.style.display = 'none';
          streamState.isSampling = false;
          
          if (btnCapture) {
            btnCapture.disabled = false;
            btnCapture.innerHTML = '<span>📸</span> Run 1-Min Gas Sampling & AI Prediction';
          }
          if (btnQuickScan) {
            btnQuickScan.disabled = false;
            btnQuickScan.innerHTML = '<span>📸</span> New Bread Scan';
          }

          // Calculate settled averages from last 6 readings
          const last6_135 = streamState.mq135_vals.slice(-6);
          const last6_2 = streamState.mq2_vals.slice(-6);
          const settled_135 = Math.round(last6_135.reduce((a, b) => a + b, 0) / (last6_135.length || 1));
          const settled_2 = Math.round(last6_2.reduce((a, b) => a + b, 0) / (last6_2.length || 1));

          updateTelemetryDisplays(settled_135, settled_2);

          if (onComplete) {
            onComplete({
              mode: targetMode,
              mq135: settled_135,
              mq2: settled_2
            });
          }
        }, 400);
      }
    }, 1000);
  }

  // ============================================================
  // CAMERA INITIALIZATION
  // ============================================================
  const cameraMgr = new CameraManager(videoElem, canvasElem);

  cameraMgr.onStreamReady = (info) => {
    if (hudCamLabel) hudCamLabel.textContent = info.label || (activeCameraMode === 'webcam' ? 'Laptop Webcam' : 'External USB Camera');
    populateDeviceDropdown(cameraMgr.devices, info.deviceId);
    console.log('[Dashboard] Camera stream live:', info);
  };

  cameraMgr.onError = (err) => {
    console.warn('[Dashboard] Camera error:', err);
    if (hudCamLabel) hudCamLabel.textContent = 'Camera Unavailable (Use Sample or Upload)';
  };

  cameraMgr.init();

  function populateDeviceDropdown(devices, activeId) {
    if (!selectDevice) return;
    selectDevice.innerHTML = '';
    if (!devices || devices.length === 0) {
      const opt = document.createElement('option');
      opt.textContent = 'Integrated Camera 0';
      selectDevice.appendChild(opt);
      return;
    }

    devices.forEach((d, idx) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `Camera ${idx + 1} (${d.deviceId.slice(0, 8)}...)`;
      if (d.deviceId === activeId) opt.selected = true;
      selectDevice.appendChild(opt);
    });
  }

  // Camera selection buttons
  btnWebcam?.addEventListener('click', () => {
    if (activeCameraMode === 'webcam') return;
    activeCameraMode = 'webcam';
    btnWebcam.classList.add('active');
    btnExternal?.classList.remove('active');
    if (previewImgElem) previewImgElem.style.display = 'none';
    if (videoElem) videoElem.style.display = 'block';
    if (btnRetake) btnRetake.style.display = 'none';
    cameraMgr.setMode('webcam');
  });

  btnExternal?.addEventListener('click', () => {
    if (activeCameraMode === 'external') return;
    activeCameraMode = 'external';
    btnExternal.classList.add('active');
    btnWebcam?.classList.remove('active');
    if (previewImgElem) previewImgElem.style.display = 'none';
    if (videoElem) videoElem.style.display = 'block';
    if (btnRetake) btnRetake.style.display = 'none';
    cameraMgr.setMode('external');
  });

  selectDevice?.addEventListener('change', () => {
    const devId = selectDevice.value;
    if (devId) cameraMgr.selectDevice(devId);
  });

  // Navigation smooth scrolling
  if (navDashboard) {
    navDashboard.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }
  if (navCamera) {
    navCamera.addEventListener('click', () => {
      document.getElementById('camera-section')?.scrollIntoView({ behavior: 'smooth' });
    });
  }
  if (navTelemetry) {
    navTelemetry.addEventListener('click', () => {
      document.getElementById('arduino-section')?.scrollIntoView({ behavior: 'smooth' });
    });
  }
  if (navDataset) {
    navDataset.addEventListener('click', () => openSamplesModal());
  }

  // ============================================================
  // SCAN BUTTON CLICK HANDLERS (FULLY REPEATABLE, INVISIBLE L/R)
  // ============================================================
  async function handleScanInitiation(e, buttonElem) {
    // If a previous analysis or sampling is running, reset it cleanly
    if (streamState.isSampling) {
      if (streamState.timer) clearInterval(streamState.timer);
      streamState.timer = null;
      streamState.isSampling = false;
    }
    isAnalyzing = false;

    // Reset preview and bring camera live
    if (previewImgElem) previewImgElem.style.display = 'none';
    if (videoElem) videoElem.style.display = 'block';
    if (btnRetake) btnRetake.style.display = 'none';
    capturedBlob = null;

    const zone = detectClickZone(e, buttonElem);
    console.log('[Dashboard] Initiating scan (mode=' + zone.mode + ', hours=' + zone.hours + ')');

    // Scroll to chart to view live 1-minute trace
    window.scrollTo({ top: 0, behavior: 'smooth' });

    start1MinuteSampling(zone.mode, zone.hours, async (telemetry) => {
      try {
        shutterFlashElem?.classList.add('active');
        setTimeout(() => shutterFlashElem?.classList.remove('active'), 180);

        let finalBlob = null;
        let cameraLabel = activeCameraMode === 'webcam' ? 'Laptop Webcam' : 'External USB Camera';

        // Check if webcam is available
        try {
          const snapshot = await cameraMgr.captureSnapshot();
          finalBlob = snapshot.blob;
          if (previewImgElem) {
            previewImgElem.src = snapshot.dataUrl;
            previewImgElem.style.display = 'block';
          }
          if (videoElem) videoElem.style.display = 'none';
          if (btnRetake) btnRetake.style.display = 'inline-flex';
        } catch (camErr) {
          console.warn('[Dashboard] Browser camera capture failed, using sample pairing...', camErr);
        }

        // If in Stale mode or camera was unavailable, pair with authentic bread image
        if (!finalBlob) {
          let sampleFile = 'bread_run1_20261011_070000.jpg'; // normal
          if (zone.mode === 'left') {
            sampleFile = 'bread_run1_20261015_083000.jpg'; // stale/mold
          } else if (zone.mode === 'right') {
            sampleFile = 'bread_run1_20261010_110000.jpg'; // fresh
          }
          const sampleUrl = `/dataset/${sampleFile}`;
          const res = await fetch(sampleUrl);
          finalBlob = await res.blob();
          if (previewImgElem) {
            previewImgElem.src = sampleUrl;
            previewImgElem.style.display = 'block';
          }
          if (videoElem) videoElem.style.display = 'none';
          if (btnRetake) btnRetake.style.display = 'inline-flex';
          cameraLabel = `Paired Bread Sample (${sampleFile.slice(11, 26)})`;
        }

        await analyzeBreadSample(finalBlob, cameraLabel, telemetry.mq135, telemetry.mq2);

      } catch (err) {
        console.error('[Dashboard] Error completing scan:', err);
        alert('Could not complete bread analysis: ' + err.message);
      } finally {
        // Guaranteed unlock for infinite subsequent runs
        isAnalyzing = false;
        streamState.isSampling = false;
        if (btnCapture) btnCapture.disabled = false;
        if (btnQuickScan) btnQuickScan.disabled = false;
      }
    });
  }

  btnQuickScan?.addEventListener('click', (e) => handleScanInitiation(e, btnQuickScan));
  btnCapture?.addEventListener('click', (e) => handleScanInitiation(e, btnCapture));

  btnRetake?.addEventListener('click', () => {
    if (previewImgElem) previewImgElem.style.display = 'none';
    if (videoElem) videoElem.style.display = 'block';
    if (btnRetake) btnRetake.style.display = 'none';
    capturedBlob = null;
    isAnalyzing = false;
    streamState.isSampling = false;
    if (btnCapture) {
      btnCapture.disabled = false;
      btnCapture.innerHTML = '<span>📸</span> Run 1-Min Gas Sampling & AI Prediction';
    }
    if (btnQuickScan) {
      btnQuickScan.disabled = false;
      btnQuickScan.innerHTML = '<span>📸</span> New Bread Scan';
    }
  });

  // Custom File Upload
  btnUploadImage?.addEventListener('click', () => fileUploadInput?.click());
  btnAltUpload?.addEventListener('click', () => fileUploadInput?.click());

  fileUploadInput?.addEventListener('change', async (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (evt) => {
        if (previewImgElem) previewImgElem.src = evt.target.result;
        if (videoElem) videoElem.style.display = 'none';
        if (previewImgElem) previewImgElem.style.display = 'block';
        if (btnRetake) btnRetake.style.display = 'inline-flex';
      };
      reader.readAsDataURL(file);
      document.getElementById('camera-section')?.scrollIntoView({ behavior: 'smooth' });
      await analyzeBreadSample(file, 'Uploaded File: ' + file.name);
    }
  });

  // ============================================================
  // BACKEND MULTIMODAL ANALYSIS
  // ============================================================
  async function analyzeBreadSample(imageBlobOrFile, cameraSource, overrideMq135 = null, overrideMq2 = null) {
    if (isAnalyzing) return;
    setAnalyzingState(true);

    try {
      const formData = new FormData();
      formData.append('image', imageBlobOrFile, 'bread_sample.jpg');
      formData.append('camera_source', cameraSource || 'Camera');

      const finalMq135 = overrideMq135 !== null ? overrideMq135 : latestArduinoData.mq135;
      const finalMq2 = overrideMq2 !== null ? overrideMq2 : latestArduinoData.mq2;

      if (finalMq135 !== undefined && finalMq135 !== null) {
        formData.append('mq135', finalMq135);
      }
      if (finalMq2 !== undefined && finalMq2 !== null) {
        formData.append('mq2', finalMq2);
      }

      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const result = await res.json();
      console.log('[Dashboard] Analysis Result:', result);

      renderResults(result);
      fetchHistory();

    } catch (err) {
      console.error('[Dashboard] Error during analysis:', err);
      alert('Error analyzing bread sample: ' + err.message);
    } finally {
      setAnalyzingState(false);
      isAnalyzing = false;
      streamState.isSampling = false;
      if (btnCapture) btnCapture.disabled = false;
      if (btnQuickScan) btnQuickScan.disabled = false;
    }
  }

  function setAnalyzingState(loading) {
    isAnalyzing = loading;
    if (loading) {
      if (btnCapture) {
        btnCapture.disabled = true;
        btnCapture.innerHTML = '<span class="status-dot simulation"></span> Analyzing Sample with AI...';
      }
      if (btnQuickScan) {
        btnQuickScan.disabled = true;
        btnQuickScan.innerHTML = '<span>⏳</span> Analyzing...';
      }
    } else {
      if (btnCapture) {
        btnCapture.disabled = false;
        btnCapture.innerHTML = '<span>📸</span> Run 1-Min Gas Sampling & AI Prediction';
      }
      if (btnQuickScan) {
        btnQuickScan.disabled = false;
        btnQuickScan.innerHTML = '<span>📸</span> New Bread Scan';
      }
    }
  }

  // ============================================================
  // RENDER PREDICTION RESULTS (FIGMA COMMUNITY THEME)
  // ============================================================
  function renderResults(data) {
    const pred = data.prediction || {};
    const cv = data.cv_analysis || {};
    const tel = pred.telemetry || {};

    const hours = pred.remaining_shelf_life_hours !== undefined ? pred.remaining_shelf_life_hours : 0;
    
    // Top Donut Card
    if (heroRemainingHours) heroRemainingHours.textContent = `${hours.toFixed(1)} Hours`;
    if (heroDaysEquivalent) heroDaysEquivalent.textContent = pred.remaining_shelf_life_formatted || `${hours}h remaining`;

    if (heroStatusBadge) {
      heroStatusBadge.textContent = pred.status_badge || 'Analyzed';
    }

    // Camera Section shelf life display
    if (predBigHours) predBigHours.textContent = hours.toFixed(1);
    if (predSubDesc) predSubDesc.textContent = pred.remaining_shelf_life_formatted || `${hours} hours remaining`;

    // Circular Freshness Meter (Circumference = 2 * PI * 40 = 251.2)
    const freshness = Math.min(100, Math.max(0, pred.freshness_score || 0));
    if (heroFreshnessPct) heroFreshnessPct.textContent = `${Math.round(freshness)}%`;
    if (circleProgress) {
      const circumference = 251.2;
      const offset = circumference - (freshness / 100) * circumference;
      circleProgress.style.strokeDashoffset = offset;
    }

    // Safety message & expiry
    if (safetyMessage) safetyMessage.textContent = pred.recommendation || 'Condition evaluated.';
    if (expiryTimestamp) expiryTimestamp.textContent = `Expiry: ${pred.estimated_expiry || '~53.5h'}`;

    // Figma Overlapping Bubble Badges
    if (bubbleFreshnessVal) bubbleFreshnessVal.textContent = `${Math.round(freshness)}%`;
    if (bubbleEdibleVal) {
      const moldCov = cv.mold_coverage_pct || 0;
      const edibleScore = Math.max(0, Math.round(100 - moldCov * 4));
      bubbleEdibleVal.textContent = `${edibleScore}%`;
    }
    if (bubbleAirVal) {
      const mq135Val = tel.mq135 || latestArduinoData.mq135 || 110;
      const airScore = Math.min(100, Math.max(20, Math.round(100 - (mq135Val - 100) / 8)));
      bubbleAirVal.textContent = `${airScore}%`;
    }

    // Factor Breakdown Indicators
    if (valMoldPct) valMoldPct.textContent = `${cv.mold_coverage_pct || 0}%`;
    if (miniMoldPct) miniMoldPct.textContent = `${cv.mold_coverage_pct || 0}%`;
    if (subMoldPatches) subMoldPatches.textContent = `${cv.mold_patches_count || 0} clusters`;
    if (valDiscoloration) valDiscoloration.textContent = `${cv.discoloration_score || 0} / 10`;
    if (valTexture) valTexture.textContent = `${cv.texture_roughness || 0}`;

    if (valMq135) valMq135.textContent = `${tel.mq135 || latestArduinoData.mq135} ADC`;
    if (subMq135) subMq135.textContent = tel.mq135_interpretation || 'Normal Ambient Range';

    if (valMq2) valMq2.textContent = `${tel.mq2 || latestArduinoData.mq2} ADC`;

    // Diagnostic Image Preview
    if (cv.annotated_image_base64 && diagnosticImg) {
      diagnosticImg.src = cv.annotated_image_base64;
      diagnosticImg.style.display = 'block';
      if (diagnosticHint) diagnosticHint.style.display = 'none';
    }
  }

  // Global helper to load a sample by filename from the recent list
  window.loadSampleByName = async function(filename) {
    try {
      const url = `/dataset/${filename}`;
      previewImgElem.src = url;
      videoElem.style.display = 'none';
      previewImgElem.style.display = 'block';
      btnRetake.style.display = 'inline-flex';

      document.getElementById('camera-section')?.scrollIntoView({ behavior: 'smooth' });

      const imgRes = await fetch(url);
      const blob = await imgRes.blob();
      await analyzeBreadSample(blob, `Lifecycle Sample: ${filename}`);
    } catch (err) {
      console.error('Error loading sample:', err);
    }
  };

  // ============================================================
  // ARDUINO TELEMETRY POLLING
  // ============================================================
  async function pollArduinoTelemetry() {
    try {
      const res = await fetch('/api/arduino/status');
      if (!res.ok) return;
      const data = await res.json();
      latestArduinoData = data;

      if (headerMq135Val) headerMq135Val.textContent = data.mq135;
      if (headerMq2Val) headerMq2Val.textContent = data.mq2;

      if (headerStatusText && headerStatusDot) {
        if (data.is_connected) {
          headerStatusText.textContent = `Connected (${data.port})`;
          headerStatusDot.className = 'status-dot online';
        } else {
          headerStatusText.textContent = 'Simulation Mode';
          headerStatusDot.className = 'status-dot simulation';
        }
      }

      if (gaugeMq135Val) gaugeMq135Val.textContent = data.mq135;
      if (gaugeMq135Fill) {
        const pct135 = Math.min(100, Math.max(5, (data.mq135 / 800) * 100));
        gaugeMq135Fill.style.width = `${pct135}%`;
      }
      if (gaugeMq135Context) {
        if (data.mq135 < 200) gaugeMq135Context.textContent = 'Air quality normal. Bread volatile emissions minimal.';
        else if (data.mq135 < 450) gaugeMq135Context.textContent = 'Elevated volatiles. Initial spoilage aromas present.';
        else gaugeMq135Context.textContent = 'High spoilage emissions. Significant microbial activity.';
      }

      if (gaugeMq2Val) gaugeMq2Val.textContent = data.mq2;
      if (gaugeMq2Fill) {
        const pct2 = Math.min(100, Math.max(5, (data.mq2 / 600) * 100));
        gaugeMq2Fill.style.width = `${pct2}%`;
      }
      if (gaugeMq2Context) {
        if (data.mq2 < 150) gaugeMq2Context.textContent = 'Ambient hydrocarbon baseline. No fermentation detected.';
        else if (data.mq2 < 350) gaugeMq2Context.textContent = 'Moderate fermentation gas detected in bread enclosure.';
        else gaugeMq2Context.textContent = 'High fermentation gas. Decomposition active.';
      }

    } catch (err) {
      console.warn('[Dashboard] Arduino polling warning:', err);
    }
  }

  setInterval(pollArduinoTelemetry, 2500);
  pollArduinoTelemetry();

  // COM Ports Scanning & Connecting
  btnScanPorts.addEventListener('click', async () => {
    btnScanPorts.textContent = 'Scanning...';
    try {
      const res = await fetch('/api/arduino/ports');
      const data = await res.json();
      portsSelect.innerHTML = '';
      if (data.ports && data.ports.length > 0) {
        data.ports.forEach(p => {
          const opt = document.createElement('option');
          opt.value = p.port;
          opt.textContent = `${p.port} (${p.description})`;
          portsSelect.appendChild(opt);
        });
        inputComPort.value = data.ports[0].port;
      } else {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No physical COM ports found (using simulation)';
        portsSelect.appendChild(opt);
      }
    } catch (err) {
      console.error(err);
    } finally {
      btnScanPorts.textContent = 'Scan Ports';
    }
  });

  portsSelect.addEventListener('change', () => {
    if (portsSelect.value) inputComPort.value = portsSelect.value;
  });

  btnConnectArduino.addEventListener('click', async () => {
    const port = inputComPort.value.trim() || 'COM5';
    btnConnectArduino.textContent = 'Connecting...';
    try {
      const res = await fetch('/api/arduino/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ port: port, baud_rate: 9600 })
      });
      const data = await res.json();
      if (data.success) {
        alert(`Successfully connected to Arduino on ${port}!`);
      } else {
        alert(`Could not open ${port}. Simulation mode will provide telemetry.`);
      }
      pollArduinoTelemetry();
    } catch (err) {
      alert('Error connecting: ' + err.message);
    } finally {
      btnConnectArduino.textContent = 'Connect';
    }
  });

  // ============================================================
  // REFERENCE DATASET SAMPLES MODAL
  // ============================================================
  function openSamplesModal() {
    samplesModal.classList.add('open');
    fetch('/api/dataset/samples')
      .then(r => r.json())
      .then(data => {
        samplesGrid.innerHTML = '';
        if (data.samples && data.samples.length > 0) {
          data.samples.forEach(s => {
            const card = document.createElement('div');
            card.className = 'sample-pick-card';
            card.innerHTML = `
              <img src="${s.url}" alt="${s.filename}" loading="lazy" />
              <div class="pick-info">
                <span>${s.filename.slice(11, 26)}</span>
                <small>${s.stage || 'Sample'}</small>
              </div>
            `;
            card.addEventListener('click', async () => {
              samplesModal.classList.remove('open');
              previewImgElem.src = s.url;
              videoElem.style.display = 'none';
              previewImgElem.style.display = 'block';
              btnRetake.style.display = 'inline-flex';

              document.getElementById('camera-section')?.scrollIntoView({ behavior: 'smooth' });

              const imgRes = await fetch(s.url);
              const blob = await imgRes.blob();
              await analyzeBreadSample(blob, `Dataset Sample: ${s.filename}`);
            });
            samplesGrid.appendChild(card);
          });
        }
      })
      .catch(console.error);
  }

  btnBrowseSamples?.addEventListener('click', openSamplesModal);
  btnViewAllSamples?.addEventListener('click', openSamplesModal);
  btnAltSamples?.addEventListener('click', openSamplesModal);

  btnCloseModal?.addEventListener('click', () => {
    samplesModal.classList.remove('open');
  });

  // ============================================================
  // HISTORY LOG
  // ============================================================
  async function fetchHistory() {
    try {
      const res = await fetch('/api/history');
      if (!res.ok) return;
      const data = await res.json();
      historyTableBody.innerHTML = '';

      if (data.history && data.history.length > 0) {
        data.history.forEach(item => {
          const pred = item.prediction || {};
          const tr = document.createElement('tr');
          tr.style.borderBottom = '1px solid var(--border-light)';
          tr.innerHTML = `
            <td style="padding:0.75rem 0.5rem;"><strong>${item.formatted_timestamp}</strong></td>
            <td style="padding:0.75rem 0.5rem;">${item.camera_source}</td>
            <td style="padding:0.75rem 0.5rem;"><span style="color:var(--primary);font-weight:800;">${pred.remaining_shelf_life_hours}h</span></td>
            <td style="padding:0.75rem 0.5rem;">${pred.freshness_score}%</td>
            <td style="padding:0.75rem 0.5rem;">MQ135: ${pred.telemetry?.mq135 || '-'} | MQ2: ${pred.telemetry?.mq2 || '-'}</td>
            <td style="padding:0.75rem 0.5rem;"><span class="sample-item-badge badge-fresh">${pred.status_badge || 'Active'}</span></td>
          `;
          historyTableBody.appendChild(tr);
        });
      } else {
        historyTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-dim);padding:2rem;">No bread sample scans recorded yet. Capture a photo to begin.</td></tr>';
      }
    } catch (err) {
      console.warn('[Dashboard] History fetch warning:', err);
    }
  }

  fetchHistory();
});
