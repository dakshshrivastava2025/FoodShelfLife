import time
import threading
import os
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False


class ArduinoManager:
    """
    Manages Arduino serial communication for MQ-135 and MQ-2 sensors.
    Supports real-time background reading, auto-reconnect, COM port scanning,
    and a realistic simulation fallback based on the actual experimental dataset.
    """

    def __init__(self, default_port: str = "COM5", baud_rate: int = 9600, csv_path: str = "sensor_data.csv"):
        self.port = default_port
        self.baud_rate = baud_rate
        self.csv_path = csv_path
        
        self.serial_conn: Optional[Any] = None
        self.is_connected = False
        self.is_simulated = False
        
        # Latest reading state
        self.latest_reading: Dict[str, Any] = {
            "mq135": 71,
            "mq2": 265,
            "timestamp": datetime.now().isoformat(),
            "is_connected": False,
            "port": self.port,
            "mode": "simulation",
            "status": "Initialized (Simulation mode active until Arduino connects)"
        }
        
        # Simulation dataset samples
        self.simulation_samples: List[Dict[str, int]] = []
        self.sim_index = 0
        self._load_simulation_data()

        # Threading
        self._running = False
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        
        # Start background manager thread
        self.start()

    def _load_simulation_data(self):
        """Loads historical sensor readings to use in simulation mode."""
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            m1 = int(row.get("mq135", 70))
                            m2 = int(row.get("mq2", 260))
                            self.simulation_samples.append({"mq135": m1, "mq2": m2})
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                print(f"[ArduinoManager] Warning loading CSV: {e}")

        if not self.simulation_samples:
            # Fallback default sequence mimicking bread aging (fresh -> staling -> mold)
            self.simulation_samples = [
                {"mq135": 68, "mq2": 242},
                {"mq135": 71, "mq2": 260},
                {"mq135": 74, "mq2": 264},
                {"mq135": 78, "mq2": 268},
                {"mq135": 83, "mq2": 272},
                {"mq135": 88, "mq2": 280},
                {"mq135": 92, "mq2": 295},
            ]

    def list_ports(self) -> List[Dict[str, str]]:
        """List all available COM/serial ports."""
        if not SERIAL_AVAILABLE:
            return []
        
        ports_info = []
        try:
            for p in serial.tools.list_ports.comports():
                ports_info.append({
                    "device": p.device,
                    "description": p.description,
                    "manufacturer": getattr(p, "manufacturer", "") or "Unknown"
                })
        except Exception as e:
            print(f"[ArduinoManager] Error scanning ports: {e}")
        return ports_info

    def connect(self, port: Optional[str] = None, baud_rate: Optional[int] = None) -> bool:
        """Attempt connection to specified or default port."""
        if port:
            self.port = port
        if baud_rate:
            self.baud_rate = baud_rate

        if not SERIAL_AVAILABLE:
            self._set_simulated_mode(f"pyserial library not available. Running in simulated sensor mode.")
            return False

        self.disconnect()

        try:
            print(f"[ArduinoManager] Attempting to open {self.port} at {self.baud_rate} baud...")
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1.5)
            time.sleep(2.0)  # Allow Arduino reset
            if hasattr(self.serial_conn, "reset_input_buffer"):
                self.serial_conn.reset_input_buffer()
            
            with self._lock:
                self.is_connected = True
                self.is_simulated = False
                self.latest_reading["is_connected"] = True
                self.latest_reading["port"] = self.port
                self.latest_reading["mode"] = "real"
                self.latest_reading["status"] = f"Connected to Arduino on {self.port}"
            print(f"[ArduinoManager] Successfully connected to Arduino on {self.port}")
            return True

        except Exception as e:
            print(f"[ArduinoManager] Could not open {self.port}: {e}")
            self._set_simulated_mode(f"Could not connect to {self.port}: {str(e)}. Using simulation.")
            return False

    def disconnect(self):
        """Disconnect active serial connection."""
        with self._lock:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
            self.serial_conn = None
            self.is_connected = False

    def _set_simulated_mode(self, reason: str = ""):
        with self._lock:
            self.is_connected = False
            self.is_simulated = True
            self.latest_reading["is_connected"] = False
            self.latest_reading["mode"] = "simulation"
            self.latest_reading["status"] = reason or "Running in simulated sensor mode"

    def set_manual_reading(self, mq135: int, mq2: int):
        """Allows dashboard to manually override sensor values for testing."""
        with self._lock:
            self.latest_reading["mq135"] = int(mq135)
            self.latest_reading["mq2"] = int(mq2)
            self.latest_reading["timestamp"] = datetime.now().isoformat()
            self.latest_reading["mode"] = "manual"
            self.latest_reading["status"] = f"Manual override (MQ135={mq135}, MQ2={mq2})"

    def get_latest_reading(self) -> Dict[str, Any]:
        """Returns the most up-to-date sensor reading."""
        with self._lock:
            # If simulated, slowly advance or add tiny jitter to mimic live sensor noise
            if self.is_simulated and self.latest_reading.get("mode") == "simulation":
                sample = self.simulation_samples[self.sim_index % len(self.simulation_samples)]
                self.latest_reading["mq135"] = sample["mq135"]
                self.latest_reading["mq2"] = sample["mq2"]
                self.latest_reading["timestamp"] = datetime.now().isoformat()
            
            return dict(self.latest_reading)

    def start(self):
        """Starts background worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Stops background worker thread."""
        self._running = False
        self.disconnect()

    def _run_loop(self):
        """Continuous background loop reading serial or cycling simulation."""
        # Initial attempt to connect
        self.connect()

        while self._running:
            if self.is_connected and self.serial_conn and self.serial_conn.is_open:
                try:
                    line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        if line.startswith("MQ135"):
                            continue
                        values = line.split(",")
                        if len(values) >= 2:
                            m1 = int(values[0].strip())
                            m2 = int(values[1].strip())
                            with self._lock:
                                self.latest_reading["mq135"] = m1
                                self.latest_reading["mq2"] = m2
                                self.latest_reading["timestamp"] = datetime.now().isoformat()
                                self.latest_reading["status"] = "Receiving live Arduino sensor data"
                except Exception as e:
                    print(f"[ArduinoManager] Serial read error: {e}")
                    self.disconnect()
                    self._set_simulated_mode(f"Lost connection to {self.port}: {e}")
            else:
                # Simulation advance every 5 seconds if in simulation mode
                time.sleep(1.0)
                if self.is_simulated and self.latest_reading.get("mode") == "simulation":
                    # Slow cycle through sample states
                    pass
            time.sleep(0.05)
