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
  btnWebcam.addEventListener('click', () => {
    if (activeCameraMode === 'webcam') return;
    activeCameraMode = 'webcam';
    btnWebcam.classList.add('active');
    btnExternal.classList.remove('active');
    previewImgElem.style.display = 'none';
    videoElem.style.display = 'block';
    btnRetake.style.display = 'none';
    cameraMgr.setMode('webcam');
  });

  btnExternal.addEventListener('click', () => {
    if (activeCameraMode === 'external') return;
    activeCameraMode = 'external';
    btnExternal.classList.add('active');
    btnWebcam.classList.remove('active');
    previewImgElem.style.display = 'none';
    videoElem.style.display = 'block';
    btnRetake.style.display = 'none';
    cameraMgr.setMode('external');
  });

  selectDevice.addEventListener('change', () => {
    const devId = selectDevice.value;
    if (devId) cameraMgr.selectDevice(devId);
  });

  // Navigation smooth scrolling
  if (btnQuickScan) {
    btnQuickScan.addEventListener('click', () => {
      document.getElementById('camera-section')?.scrollIntoView({ behavior: 'smooth' });
    });
  }
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
  // SHUTTER & PHOTO CAPTURE
  // ============================================================
  btnCapture.addEventListener('click', async () => {
    if (isAnalyzing) return;

    try {
      // Trigger Shutter flash effect
      shutterFlashElem.classList.add('active');
      setTimeout(() => shutterFlashElem.classList.remove('active'), 180);

      // Grab snapshot from active camera stream
      const snapshot = await cameraMgr.captureSnapshot();
      capturedBlob = snapshot.blob;

      // Show preview
      previewImgElem.src = snapshot.dataUrl;
      videoElem.style.display = 'none';
      previewImgElem.style.display = 'block';
      btnRetake.style.display = 'inline-flex';

      // Send to backend for analysis
      await analyzeBreadSample(capturedBlob, activeCameraMode === 'webcam' ? 'Laptop Webcam' : 'External USB Camera');

    } catch (err) {
      console.warn('[Dashboard] Browser stream capture failed, trying backend camera...', err);
      try {
        setAnalyzingState(true);
        const camIdx = activeCameraMode === 'webcam' ? 0 : 1;
        const res = await fetch(`/api/cameras/capture?camera_index=${camIdx}`, { method: 'POST' });
        if (!res.ok) throw new Error('Backend camera capture failed');
        const data = await res.json();
        
        const imgRes = await fetch(data.url);
        const blob = await imgRes.blob();
        await analyzeBreadSample(blob, `Host Camera ${camIdx}`);
      } catch (backendErr) {
        alert('Camera capture unavailable. You can click "Browse Dataset" or "Upload Photo" to test.');
      } finally {
        setAnalyzingState(false);
      }
    }
  });

  btnRetake.addEventListener('click', () => {
    previewImgElem.style.display = 'none';
    videoElem.style.display = 'block';
    btnRetake.style.display = 'none';
    capturedBlob = null;
  });

  // Custom File Upload
  btnUploadImage?.addEventListener('click', () => fileUploadInput.click());
  btnAltUpload?.addEventListener('click', () => fileUploadInput.click());

  fileUploadInput.addEventListener('change', async (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (evt) => {
        previewImgElem.src = evt.target.result;
        videoElem.style.display = 'none';
        previewImgElem.style.display = 'block';
        btnRetake.style.display = 'inline-flex';
      };
      reader.readAsDataURL(file);
      document.getElementById('camera-section')?.scrollIntoView({ behavior: 'smooth' });
      await analyzeBreadSample(file, 'Uploaded File: ' + file.name);
    }
  });

  // ============================================================
  // BACKEND MULTIMODAL ANALYSIS
  // ============================================================
  async function analyzeBreadSample(imageBlobOrFile, cameraSource) {
    if (isAnalyzing) return;
    setAnalyzingState(true);

    try {
      const formData = new FormData();
      formData.append('image', imageBlobOrFile, 'bread_sample.jpg');
      formData.append('camera_source', cameraSource || 'Camera');

      if (latestArduinoData.mq135 && latestArduinoData.mq2) {
        formData.append('mq135', latestArduinoData.mq135);
        formData.append('mq2', latestArduinoData.mq2);
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
    }
  }

  function setAnalyzingState(loading) {
    isAnalyzing = loading;
    if (loading) {
      btnCapture.disabled = true;
      btnCapture.innerHTML = '<span class="status-dot simulation"></span> Analyzing Sample with AI...';
    } else {
      btnCapture.disabled = false;
      btnCapture.innerHTML = '<span>📸</span> Click Photo & Predict Shelf Life';
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
