import cv2
import serial
import csv
import os
import time
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

ARDUINO_PORT = "COM5"
BAUD_RATE = 9600

# TEST MODE:
CAPTURE_INTERVAL_SECONDS = 10

# For real experiment, change to:
# CAPTURE_INTERVAL_SECONDS = 10 * 60

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

file_exists = os.path.exists(
    CSV_FILE
)

csv_file = open(
    CSV_FILE,
    mode="a",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(
    csv_file
)

if not file_exists:

    csv_writer.writerow([
        "sample_number",
        "timestamp",
        "hours_since_start",
        "mq135",
        "mq2",
        "image_filename",
        "camera_index"
    ])

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
# START EXPERIMENT
# ============================================================

experiment_start = datetime.now()

print("\n========================================")
print("BREAD DATA COLLECTION STARTED")
print("========================================")
print(f"Run              : {RUN_NAME}")
print(f"Arduino          : {ARDUINO_PORT}")
print(f"USB Camera       : {CAMERA_INDEX}")
print(
    f"Capture interval : "
    f"{CAPTURE_INTERVAL_SECONDS} seconds"
)
print(f"CSV              : {CSV_FILE}")
print(f"Images           : {IMAGE_FOLDER}")
print("========================================")
print("Press Q in camera window to stop.")
print("Or press CTRL+C in terminal.")
print("========================================\n")


# ============================================================
# MAIN LOOP
# ============================================================

sample_number = 1
last_capture_time = 0

try:

    while True:

        # --------------------------------------------
        # Keep camera preview alive continuously
        # --------------------------------------------

        show_live_preview()

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print(
                "\nStopping data collection..."
            )
            break

        current_time = time.time()

        # --------------------------------------------
        # Capture only when interval has passed
        # --------------------------------------------

        if (
            current_time - last_capture_time
            >= CAPTURE_INTERVAL_SECONDS
        ):

            cycle_start = time.time()

            now = datetime.now()

            timestamp = now.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            filename_timestamp = (
                now.strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            hours_since_start = (
                now - experiment_start
            ).total_seconds() / 3600

            # ----------------------------------------
            # READ GAS SENSORS
            # ----------------------------------------

            mq135, mq2 = read_sensor()

            # ----------------------------------------
            # IMAGE FILENAME
            # ----------------------------------------

            image_filename = (
                f"bread_{RUN_NAME}_"
                f"{filename_timestamp}.jpg"
            )

            # ----------------------------------------
            # CAPTURE IMAGE
            # ----------------------------------------

            image_saved = capture_image(
                image_filename
            )

            if not image_saved:
                image_filename = "CAMERA_ERROR"

            # ----------------------------------------
            # SAVE CSV
            # ----------------------------------------

            csv_writer.writerow([
                sample_number,
                timestamp,
                round(
                    hours_since_start,
                    4
                ),
                mq135,
                mq2,
                image_filename,
                CAMERA_INDEX
            ])

            csv_file.flush()

            # ----------------------------------------
            # TERMINAL OUTPUT
            # ----------------------------------------

            print(
                "\n----------------------------------------"
            )

            print(
                f"Sample        : {sample_number}"
            )

            print(
                f"Time          : {timestamp}"
            )

            print(
                f"Hours elapsed : "
                f"{hours_since_start:.4f}"
            )

            print(
                f"MQ135         : {mq135}"
            )

            print(
                f"MQ2           : {mq2}"
            )

            print(
                f"Image         : {image_filename}"
            )

            print(
                "----------------------------------------\n"
            )

            sample_number += 1

            last_capture_time = current_time

        # Reduce CPU usage slightly
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