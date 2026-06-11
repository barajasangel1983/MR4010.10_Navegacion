# ============================================================
# dataset_mode_v3 — Behavioral Cloning Dataset Module
# ============================================================
#
# PURPOSE
#   Capture synchronized camera frames + steering/speed/brake values
#   for training a behavioral cloning model (end-to-end driving).
#
# TRIGGER
#   PS4 Triangle (btn 2) or Keyboard 'D' / 'd' → toggle on/off.
#   Programmatic: set_dataset_mode(True/False).
#
# OUTPUT
#   data/behavioral_dataset/images/frame_XXXXXX.jpg   (JPEG, 620×320)
#   data/behavioral_dataset/measurements.csv           (CSV with metadata)
#
# CAPTURE RATE
#   2 Hz — one sample every 0.5 seconds while dataset mode is active.
#
# CSV FORMAT (measurements.csv)
#   timestamp,image_filename,steering_angle,speed,brake
#   2026-06-11T14:32:01.234,frame_000001.jpg,0.123,45.0,0.00
#
# USAGE
#   from dataset_mode_v3 import try_capture, set_dataset_mode, toggle_dataset
#   ds_counter, ds_overlay = try_capture(frame, angle, speed, brake, current_time)
# ============================================================

import os
import csv
import time
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAPTURE_INTERVAL = 0.5       # seconds between captures (2 Hz)
JPEG_QUALITY = 85            # JPEG quality 1-100
IMG_WIDTH = 620              # Camera width
IMG_HEIGHT = 320             # Camera height
DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "behavioral_dataset"
)
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "measurements.csv")

# CSV header
CSV_HEADER = ["timestamp", "image_filename", "steering_angle", "speed", "brake"]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
dataset_mode = False
dataset_counter = 0
last_capture_time = 0.0
_csv_file_handle = None       # open file handle for append-mode writing
_csv_writer = None            # csv.writer instance

# ---------------------------------------------------------------------------
# Init / Cleanup
# ---------------------------------------------------------------------------
def _ensure_dirs():
    """Create dataset directories if they don't exist."""
    os.makedirs(IMAGES_DIR, exist_ok=True)


def _open_csv():
    """Open CSV in append mode. Create header if new file."""
    global _csv_file_handle, _csv_writer

    _ensure_dirs()

    file_exists = os.path.isfile(CSV_PATH)

    if not file_exists:
        # Create new CSV with header
        _csv_file_handle = open(CSV_PATH, "w", newline="")
        _csv_writer = csv.writer(_csv_file_handle)
        _csv_writer.writerow(CSV_HEADER)
        _csv_file_handle.flush()
        print("[DS] Created new measurements.csv")
    else:
        # Append to existing CSV
        _csv_file_handle = open(CSV_PATH, "a", newline="")
        _csv_writer = csv.writer(_csv_file_handle)
        print(f"[DS] Appending to existing measurements.csv ({CSV_PATH})")

    _csv_file_handle.flush()


def _close_csv():
    """Close CSV file handle."""
    global _csv_file_handle, _csv_writer
    if _csv_file_handle and not _csv_file_handle.closed:
        _csv_file_handle.close()
        print(f"[DS] measurements.csv saved. Total samples: {dataset_counter}")


def init_dataset():
    """Initialize dataset directories. Call once at startup."""
    _ensure_dirs()


# ---------------------------------------------------------------------------
# Toggle
# ---------------------------------------------------------------------------
def set_dataset_mode(on: bool):
    """Set dataset capture mode."""
    global dataset_mode, dataset_counter, last_capture_time
    dataset_mode = on
    if on:
        dataset_counter = 0
        last_capture_time = 0.0
        _open_csv()
        print("[DS] BEHAVIORAL CLONING DATASET MODE: ON (2 Hz)")
    else:
        _close_csv()
        print("[DS] BEHAVIORAL CLONING DATASET MODE: OFF")


def toggle_dataset():
    """Toggle dataset capture mode on/off."""
    set_dataset_mode(not dataset_mode)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def try_capture(frame, angle: float, speed: float, brake: float, current_time: float):
    """
    Attempt to capture one dataset sample if enough time has passed.

    Parameters
    ----------
    frame : np.ndarray
        BGR image from Webots camera.
    angle : float
        Steering angle (radians).
    speed : float
        Speed percentage (0-100).
    brake : float
        Brake intensity (0-1).
    current_time : float
        Current time from time.time().

    Returns
    -------
    (counter, overlay) — counter is always int, overlay is None (no overlay needed).
    """
    global dataset_counter, last_capture_time

    overlay = None

    # If dataset mode is off, nothing to do
    if not dataset_mode:
        return dataset_counter, overlay

    # Check capture interval
    if (current_time - last_capture_time) < CAPTURE_INTERVAL:
        return dataset_counter, overlay

    # Capture!
    dataset_counter += 1
    last_capture_time = current_time
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(current_time))
    # Fractional seconds
    frac = f".{int((current_time % 1) * 1000):03d}"
    timestamp = ts + frac

    # Generate filename
    filename = f"frame_{dataset_counter:06d}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)

    # Resize to target resolution (just in case camera is configured differently)
    capture_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # Save as JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    cv2.imwrite(filepath, capture_frame, encode_params)

    # Log measurement to CSV
    if _csv_writer:
        _csv_writer.writerow([timestamp, filename, f"{angle:.4f}", f"{speed:.2f}", f"{brake:.4f}"])
        # Flush after every write so data is never lost
        _csv_file_handle.flush()

    # Print every 10 captures for visibility without spam
    if dataset_counter % 10 == 0:
        print(f"[DS] Sample #{dataset_counter} | angle={angle:.3f} speed={speed:.1f} brake={brake:.2f}")

    return dataset_counter, overlay


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def get_status():
    """Return current dataset status info."""
    return {
        "mode": dataset_mode,
        "samples": dataset_counter,
        "path": IMAGES_DIR,
        "csv": CSV_PATH,
    }


# ---------------------------------------------------------------------------
# Module-level init
# ---------------------------------------------------------------------------
init_dataset()
