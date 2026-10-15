"""Collect real serial readings to CSV. Camera preview is optional and RAM-only."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

FIELDS = ['sample_number', 'timestamp', 'hours_since_start', 'mq135', 'mq2',
          'image_filename', 'camera_index', 'source']


def parse_reading(line):
    parts = line.strip().split(',')
    if len(parts) != 2:
        raise ValueError('Expected mq135,mq2')
    values = tuple(int(part) for part in parts)
    if any(value < 0 or value > 1023 for value in values):
        raise ValueError('ADC values must be between 0 and 1023')
    return values


def collect(args):
    if args.interval <= 0 or args.samples < 0:
        raise ValueError('Interval must be positive; sample count must be nonnegative')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    number, offset = 1, 0.0
    exists = output.exists() and output.stat().st_size > 0
    if exists:
        with output.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError('Existing CSV has a different schema; choose another --output')
            rows = list(reader)
            if rows:
                number = int(rows[-1]['sample_number']) + 1
                offset = float(rows[-1]['hours_since_start']) + args.interval / 3600
    camera, device = None, None
    try:
        if not args.simulate:
            import serial
            device = serial.Serial(args.port, args.baud, timeout=1)
            time.sleep(2)
            device.reset_input_buffer()
        if args.preview:
            import cv2
            camera = cv2.VideoCapture(args.camera)
            if not camera.isOpened():
                raise RuntimeError(f'Camera {args.camera} unavailable')
        started = time.monotonic()
        due = started
        if args.clock_aligned:
            due += (-time.time()) % args.interval
        written = 0
        with output.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
                handle.flush()
            print(f'Writing {output}; source={"simulated" if args.simulate else "serial"}; images are never saved', flush=True)
            while not args.samples or written < args.samples:
                if camera is not None:
                    ok, frame = camera.read()
                    if ok:
                        cv2.imshow('Bread preview - RAM only; Q to stop', frame)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                if time.monotonic() < due:
                    time.sleep(.02)
                    continue
                if args.simulate:
                    # Explicit smoke-test mode, never silently substitutes for hardware.
                    values = (80 + written % 5, 250 + written % 7)
                else:
                    device.reset_input_buffer()  # Read current telemetry, not a backlog from the wait.
                    line = device.readline().decode('utf-8', errors='replace')
                    try:
                        values = parse_reading(line)
                    except ValueError:
                        continue
                now = datetime.now().astimezone()
                writer.writerow(dict(sample_number=number, timestamp=now.isoformat(timespec='seconds'),
                                     hours_since_start=round(offset + (time.monotonic() - started) / 3600, 6),
                                     mq135=values[0], mq2=values[1], image_filename='',
                                     camera_index=args.camera if camera is not None else '',
                                     source='simulated' if args.simulate else 'serial'))
                handle.flush()
                print(number, now.isoformat(timespec='seconds'), *values, flush=True)
                number += 1
                written += 1
                due += args.interval
                if due < time.monotonic():
                    due = time.monotonic() + args.interval
    except KeyboardInterrupt:
        print('Collection stopped; CSV flushed.')
    finally:
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()
        if device is not None:
            device.close()


def main(clock_aligned=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM5')
    parser.add_argument('--baud', type=int, default=9600)
    parser.add_argument('--interval', type=float, default=300, help='Seconds between CSV rows')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parent / 'data' / 'collected' / 'sensor_data.csv'))
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--simulate', action='store_true', help='Explicit synthetic smoke-test values')
    parser.add_argument('--samples', type=int, default=0, help='Stop after N rows; 0 means continuous')
    parser.add_argument('--clock-aligned', action='store_true', default=clock_aligned)
    collect(parser.parse_args())


if __name__ == '__main__':
    main()
