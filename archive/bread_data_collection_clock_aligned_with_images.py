import cv2
import serial
import csv
import os
import time
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

ARDUINO_PORT = "COM5"
BAUD_RATE = 9600

# Clock-aligned schedule (local computer time):
# Gas: 00, 05, 10, ... 55 minutes of every hour.
# Photos: 00 and 30 minutes of every hour.
GAS_INTERVAL_MINUTES = 5
IMAGE_INTERVAL_MINUTES = 30

RUN_NAME = "run1"

BASE_FOLDER = "bread_dataset"
RUN_FOLDER = os.path.join(BASE_FOLDER, RUN_NAME)
IMAGE_FOLDER = os.path.join(RUN_FOLDER, "images")
CSV_FILE = os.path.join(RUN_FOLDER, "sensor_data.csv")

# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ============================================================
# CAMERA SELECTION
# ============================================================

def select_camera(max_cameras=6):

    print("\n========================================")
    print("WEBCAM SELECTION")
    print("========================================")

    backends = [
        ("MSMF", cv2.CAP_MSMF),
        ("DSHOW", cv2.CAP_DSHOW)
    ]

    for index in range(max_cameras):

        print(f"\nChecking camera index {index}...")

        camera_found = False

        for backend_name, backend in backends:

            print(f"Trying index {index} using {backend_name}...")

            camera = cv2.VideoCapture(index, backend)

            if not camera.isOpened():

                print(
                    f"Index {index} with {backend_name}: "
                    f"could not open"
                )

                camera.release()
                continue

            # Give USB webcam time to initialise
            time.sleep(2)

            frame = None
            success = False

            # Try several times
            for attempt in range(20):

                ret, test_frame = camera.read()

                if ret and test_frame is not None:

                    frame = test_frame
                    success = True
                    break

                time.sleep(0.2)

            if not success:

                print(
                    f"Index {index} with {backend_name}: "
                    f"opened but no frame received"
                )

                camera.release()
                continue

            camera_found = True

            print(
                f"Camera {index} works using {backend_name}"
            )

            # Show live preview
            while True:

                ret, frame = camera.read()

                if not ret or frame is None:
                    print("Frame lost.")
                    break

                preview = frame.copy()

                cv2.putText(
                    preview,
                    f"Camera {index} - {backend_name}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Camera Preview",
                    preview
                )

                # Required to keep preview responsive
                cv2.waitKey(1)

                # Take input from terminal
                print(
                    "\nLook at the preview window."
                )

                choice = input(
                    "Is this your USB webcam? "
                    "[Y/N/Q]: "
                ).strip().lower()

                if choice == "y":

                    cv2.destroyAllWindows()

                    print(
                        f"\nSelected camera {index} "
                        f"using {backend_name}"
                    )

                    return camera, index

                elif choice == "q":

                    camera.release()
                    cv2.destroyAllWindows()

                    raise SystemExit(
                        "Camera selection cancelled."
                    )

                else:

                    camera.release()
                    cv2.destroyAllWindows()

                    print(
                        f"Skipping camera {index}..."
                    )

                    break

            if camera_found:
                break

    raise RuntimeError(
        "No working webcam was selected."
    )


# ============================================================
# CONNECT TO ARDUINO
# ============================================================

print("\n========================================")
print("CONNECTING TO ARDUINO")
print("========================================")

try:

    arduino = serial.Serial(
        ARDUINO_PORT,
        BAUD_RATE,
        timeout=2
    )

except serial.SerialException as error:

    print(
        f"\nERROR: Could not open {ARDUINO_PORT}"
    )

    print(error)

    print(
        "\nCheck that:"
        "\n1. Arduino is connected."
        "\n2. COM5 is correct."
        "\n3. Arduino Serial Monitor is CLOSED."
    )

    raise SystemExit


print(
    f"Arduino connected on {ARDUINO_PORT}"
)

# Arduino resets after serial connection
time.sleep(2)

arduino.reset_input_buffer()


# ============================================================
# SELECT USB CAMERA
# ============================================================

camera, CAMERA_INDEX = select_camera()

# Set preferred resolution
camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1920
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    1080
)

actual_width = int(
    camera.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

actual_height = int(
    camera.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

print(
    f"Selected camera resolution: "
    f"{actual_width} x {actual_height}"
)


# ============================================================
# CREATE / OPEN CSV
# ============================================================

CSV_COLUMNS = [
    "sample_number",
    "timestamp",          # Scheduled clock time (e.g. 11:05:00)
    "actual_read_time",   # Actual time the gas sample was received
    "mq135",
    "mq2",
    "image_filename",     # Empty if no photo is due
    "camera_index"
]

# Don't append new-format rows to an older run's incompatible CSV.
if os.path.isfile(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
    with open(CSV_FILE, newline="", encoding="utf-8") as existing:
        old_header = next(csv.reader(existing), [])
    if old_header != CSV_COLUMNS:
        CSV_FILE = os.path.join(RUN_FOLDER, "sensor_data_clock_aligned.csv")
        print(f"Existing CSV uses another format; saving to {CSV_FILE}")

# Continue sample numbering when restarting the same run.
sample_number = 1
if os.path.isfile(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
    with open(CSV_FILE, newline="", encoding="utf-8") as existing:
        reader = csv.DictReader(existing)
        if reader.fieldnames != CSV_COLUMNS:
            raise RuntimeError(f"CSV column mismatch: {CSV_FILE}")
        for row in reader:
            try:
                sample_number = max(sample_number, int(row["sample_number"]) + 1)
            except (ValueError, TypeError):
                pass

csv_file = open(CSV_FILE, mode="a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
if os.path.getsize(CSV_FILE) == 0:
    csv_writer.writerow(CSV_COLUMNS)
    csv_file.flush()


# ============================================================
# SENSOR READING FUNCTION
# ============================================================

def read_sensor():

    while True:

        try:

            line = (
                arduino
                .readline()
                .decode(
                    "utf-8",
                    errors="ignore"
                )
                .strip()
            )

            if not line:
                continue

            # Ignore Arduino header
            if line.startswith("MQ135"):
                continue

            values = line.split(",")

            if len(values) != 2:
                continue

            mq135 = int(
                values[0]
            )

            mq2 = int(
                values[1]
            )

            return mq135, mq2

        except ValueError:

            print(
                "Invalid Arduino reading. Retrying..."
            )


# ============================================================
# IMAGE CAPTURE FUNCTION
# ============================================================

def capture_image(filename):

    # Flush a few frames
    for _ in range(5):
        camera.read()

    ret, frame = camera.read()

    if not ret or frame is None:

        print(
            "ERROR: Could not capture camera image."
        )

        return False

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    success = cv2.imwrite(
        image_path,
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    return success


# ============================================================
# OPTIONAL LIVE PREVIEW
# ============================================================

def show_live_preview():

    ret, frame = camera.read()

    if not ret:
        return

    preview = frame.copy()

    cv2.putText(
        preview,
        f"USB Camera: {CAMERA_INDEX}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        preview,
        "Bread Data Collection Running",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        preview,
        "Press Q to stop",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Bread Monitoring",
        preview
    )


# ============================================================
# CLOCK-ALIGNED SCHEDULING
# ============================================================

def next_gas_boundary(now):
    """Strictly next 5-minute boundary, never a startup reading."""
    minute = (now.minute // GAS_INTERVAL_MINUTES + 1) * GAS_INTERVAL_MINUTES
    return now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)


next_gas_time = next_gas_boundary(datetime.now())

print("\n========================================")
print("BREAD DATA COLLECTION STARTED")
print("========================================")
print(f"Run              : {RUN_NAME}")
print(f"Arduino          : {ARDUINO_PORT}")
print(f"USB Camera       : {CAMERA_INDEX}")
print(f"Gas readings     : every {GAS_INTERVAL_MINUTES} min (clock-aligned)")
print(f"Photos           : every {IMAGE_INTERVAL_MINUTES} min (:00 and :30)")
print(f"Next gas sample  : {next_gas_time:%Y-%m-%d %H:%M:%S}")
print(f"CSV              : {CSV_FILE}")
print(f"Images           : {IMAGE_FOLDER}")
print("========================================")
print("Press Q in camera window to stop.")
print("Or press CTRL+C in terminal.")
print("========================================\n")


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:
        show_live_preview()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopping data collection...")
            break

        if datetime.now() >= next_gas_time:
            scheduled_time = next_gas_time
            timestamp = scheduled_time.strftime("%Y-%m-%d %H:%M:%S")

            # Discard buffered old serial samples so we read fresh data.
            arduino.reset_input_buffer()
            mq135, mq2 = read_sensor()
            actual_read_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            image_filename = ""
            if scheduled_time.minute % IMAGE_INTERVAL_MINUTES == 0:
                filename_timestamp = scheduled_time.strftime("%Y%m%d_%H%M%S")
                image_filename = f"bread_{RUN_NAME}_{filename_timestamp}.jpg"
                if not capture_image(image_filename):
                    image_filename = "CAMERA_ERROR"

            csv_writer.writerow([
                sample_number,
                timestamp,
                actual_read_time,
                mq135,
                mq2,
                image_filename,
                CAMERA_INDEX
            ])
            csv_file.flush()

            print("\n----------------------------------------")
            print(f"Sample          : {sample_number}")
            print(f"Scheduled time  : {timestamp}")
            print(f"Actual gas read : {actual_read_time}")
            print(f"MQ135           : {mq135}")
            print(f"MQ2             : {mq2}")
            print(f"Image           : {image_filename or 'Not due'}")
            print("----------------------------------------\n")

            sample_number += 1
            # Skip any clock slots missed during slow reads or interruptions;
            # never produce artificial catch-up samples for past timestamps.
            next_gas_time = next_gas_boundary(datetime.now())
            print(f"Next gas sample : {next_gas_time:%Y-%m-%d %H:%M:%S}")

        time.sleep(0.02)


except KeyboardInterrupt:

    print(
        "\nCTRL+C detected. "
        "Stopping data collection..."
    )


finally:

    print(
        "\nClosing devices..."
    )

    camera.release()

    cv2.destroyAllWindows()

    arduino.close()

    csv_file.close()

    print(
        "Camera closed."
    )

    print(
        "Arduino disconnected."
    )

    print(
        "CSV saved."
    )

    print(
        "Data collection ended safely."
    )