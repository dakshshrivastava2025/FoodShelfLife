/**
 * CameraManager
 * Controls browser WebRTC webcam / external USB camera feeds,
 * device enumeration, live stream rendering, and high-res snapshot capture.
 */
class CameraManager {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.currentStream = null;
    this.devices = [];
    this.activeDeviceId = null;
    this.selectedMode = 'webcam'; // 'webcam' (laptop/integrated) or 'external' (usb)
    this.onStreamReady = null;
    this.onError = null;
  }

  async init() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        throw new Error('MediaDevices API not supported in this browser.');
      }

      // Request initial stream to grant permissions and retrieve device labels
      await this.refreshDevices();

      // Start initial camera stream
      await this.startCamera();
      return true;
    } catch (err) {
      console.warn('[CameraManager] Could not start browser camera stream:', err);
      if (this.onError) this.onError(err);
      return false;
    }
  }

  async refreshDevices() {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      this.devices = allDevices.filter(d => d.kind === 'videoinput');
      console.log('[CameraManager] Detected video devices:', this.devices);
      return this.devices;
    } catch (err) {
      console.warn('[CameraManager] Error enumerating devices:', err);
      return [];
    }
  }

  async startCamera(preferredDeviceId = null) {
    // Stop any existing stream
    this.stopCamera();

    let constraints = {
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      },
      audio: false
    };

    if (preferredDeviceId) {
      constraints.video.deviceId = { exact: preferredDeviceId };
      this.activeDeviceId = preferredDeviceId;
    } else if (this.selectedMode === 'external') {
      // Try to pick external USB camera or back environment camera
      const external = this.devices.find(d => 
        d.deviceId !== this.activeDeviceId &&
        (d.label.toLowerCase().includes('usb') || 
         d.label.toLowerCase().includes('external') || 
         d.label.toLowerCase().includes('back'))
      ) || (this.devices.length > 1 ? this.devices[1] : null);

      if (external) {
        constraints.video.deviceId = { exact: external.deviceId };
        this.activeDeviceId = external.deviceId;
      } else {
        constraints.video.facingMode = { ideal: 'environment' };
      }
    } else {
      // Laptop Webcam / Front facing
      const integrated = this.devices.find(d => 
        d.label.toLowerCase().includes('integrated') || 
        d.label.toLowerCase().includes('front') || 
        d.label.toLowerCase().includes('internal') ||
        d.label.toLowerCase().includes('webcam')
      ) || (this.devices.length > 0 ? this.devices[0] : null);

      if (integrated) {
        constraints.video.deviceId = { exact: integrated.deviceId };
        this.activeDeviceId = integrated.deviceId;
      } else {
        constraints.video.facingMode = { ideal: 'user' };
      }
    }

    try {
      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.currentStream;
      await this.video.play();

      // Re-read devices now that permissions are guaranteed (labels will be visible)
      await this.refreshDevices();

      if (this.onStreamReady) {
        const track = this.currentStream.getVideoTracks()[0];
        const settings = track.getSettings ? track.getSettings() : {};
        const label = track.label || 'Active Camera';
        this.onStreamReady({
          label: label,
          deviceId: settings.deviceId || this.activeDeviceId,
          mode: this.selectedMode,
          resolution: `${settings.width || 1280}x${settings.height || 720}`
        });
      }
      return true;
    } catch (err) {
      console.warn('[CameraManager] Failed to start stream with constraints:', constraints, err);
      // Fallback to basic video constraint
      try {
        this.currentStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        this.video.srcObject = this.currentStream;
        await this.video.play();
        return true;
      } catch (fallbackErr) {
        if (this.onError) this.onError(fallbackErr);
        return false;
      }
    }
  }

  setMode(mode) {
    if (this.selectedMode === mode) return;
    this.selectedMode = mode;
    this.startCamera();
  }

  selectDevice(deviceId) {
    this.activeDeviceId = deviceId;
    this.startCamera(deviceId);
  }

  stopCamera() {
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(t => t.stop());
      this.currentStream = null;
    }
  }

  captureSnapshot() {
    return new Promise((resolve, reject) => {
      if (!this.video || this.video.readyState < 2) {
        reject(new Error('Video stream is not ready for capture.'));
        return;
      }

      const width = this.video.videoWidth || 1280;
      const height = this.video.videoHeight || 720;

      this.canvas.width = width;
      this.canvas.height = height;
      const ctx = this.canvas.getContext('2d');
      ctx.drawImage(this.video, 0, 0, width, height);

      this.canvas.toBlob((blob) => {
        if (blob) {
          const dataUrl = this.canvas.toDataURL('image/jpeg', 0.95);
          resolve({ blob, dataUrl });
        } else {
          reject(new Error('Failed to create image blob from canvas.'));
        }
      }, 'image/jpeg', 0.95);
    });
  }
}

window.CameraManager = CameraManager;
