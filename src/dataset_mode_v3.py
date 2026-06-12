# ============================================================
# dataset_mode_v3 — Behavioral Cloning Dataset Module
# ============================================================
#
# PURPOSE
#   Capture synchronized camera frames + control labels for training
#   a behavioral cloning model targeting vehicle-evasion maneuvers:
#   detect vehicle in left lane → evade right → return to left lane.
#
# TRIGGER
#   PS4 Triangle (btn 2) or Keyboard 'D' / 'd' → toggle on/off.
#   Programmatic: set_dataset_mode(True/False).
#
# OUTPUT
#   data/behavioral_dataset/images/frame_XXXXXX.jpg   (JPEG, 620×320)
#   data/behavioral_dataset/measurements.csv           (CSV, append mode)
#
# CAPTURE RATE
#   2 Hz — one sample every 0.5 s. Raise CAPTURE_INTERVAL for higher Hz.
#
# FRAME COUNTER
#   Resumes from the number of existing frame_*.jpg files on disk.
#   Toggling D off/on never overwrites previous captures.
#
# SESSION ID
#   Timestamp string (YYYYmmdd_HHMMSS) generated at each activation.
#   Use it to group samples by recording run for train/val splitting.
#
# CSV COLUMNS
#   session_id             — recording session identifier
#   timestamp              — ISO-8601 with milliseconds (UTC)
#   image_filename         — frame_XXXXXX.jpg
#   steering_angle         — radians (applied this frame)
#   speed_kmh              — km/h (applied this frame)
#   brake                  — 0–1 (applied this frame)
#   autonomous_mode        — 0=manual (expert), 1=PID autonomous
#   line_detected          — 0/1 yellow lane line visible this frame
#   gps_x, gps_y           — metres (blank if GPS unavailable)
#   lidar_obstacle_detected — 0/1 LiDAR sees object in 40° front arc
#   lidar_obstacle_distance — metres to closest object (blank if none)
#   evasion_phase          — 0=NORMAL, 1=EVADING, 2=RETURNING
#   obstacle_type          — VEHICLE | OBSTACLE | LIDAR_ONLY |
#                            OBJECT_AHEAD | "" (empty = clear)
#
# USAGE
#   from dataset_mode_v3 import try_capture, set_dataset_mode, toggle_dataset
#   ds_counter, _ = try_capture(
#       frame, angle, speed, brake, current_time,
#       autonomous_mode=False, line_detected=False, gps_vals=None,
#       lidar_obstacle_detected=False, lidar_obstacle_distance=None,
#       evasion_phase=0, obstacle_type=""
#   )
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

CSV_HEADER = [
    "session_id", "timestamp", "image_filename",
    "steering_angle", "speed_kmh", "brake",
    "autonomous_mode", "line_detected",
    "gps_x", "gps_y",
    "lidar_obstacle_detected", "lidar_obstacle_distance",
    "evasion_phase", "obstacle_type",
]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
dataset_mode = False
dataset_counter = 0
last_capture_time = 0.0
_session_id = ""
_csv_file_handle = None
_csv_writer = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_existing_frames():
    """Count frame_XXXXXX.jpg files already on disk so the counter never resets."""
    if not os.path.isdir(IMAGES_DIR):
        return 0
    return sum(1 for f in os.listdir(IMAGES_DIR)
               if f.startswith("frame_") and f.endswith(".jpg"))


def _ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def _open_csv():
    global _csv_file_handle, _csv_writer
    _ensure_dirs()
    file_exists = os.path.isfile(CSV_PATH)
    if not file_exists:
        _csv_file_handle = open(CSV_PATH, "w", newline="")
        _csv_writer = csv.writer(_csv_file_handle)
        _csv_writer.writerow(CSV_HEADER)
        _csv_file_handle.flush()
        print("[DS] Created new measurements.csv")
    else:
        _csv_file_handle = open(CSV_PATH, "a", newline="")
        _csv_writer = csv.writer(_csv_file_handle)
        print(f"[DS] Appending to existing measurements.csv ({CSV_PATH})")


def _close_csv():
    global _csv_file_handle, _csv_writer
    if _csv_file_handle and not _csv_file_handle.closed:
        _csv_file_handle.close()
        print(f"[DS] measurements.csv saved. Total samples: {dataset_counter}")


def init_dataset():
    """Create dataset directories. Call once at startup."""
    _ensure_dirs()


# ---------------------------------------------------------------------------
# Toggle
# ---------------------------------------------------------------------------
def set_dataset_mode(on: bool):
    global dataset_mode, dataset_counter, last_capture_time, _session_id
    dataset_mode = on
    if on:
        dataset_counter = _count_existing_frames()
        last_capture_time = 0.0
        _session_id = time.strftime("%Y%m%d_%H%M%S")
        _open_csv()
        print(f"[DS] BEHAVIORAL CLONING DATASET MODE: ON (2 Hz) | "
              f"session={_session_id} | resuming at frame #{dataset_counter}")
    else:
        _close_csv()
        print("[DS] BEHAVIORAL CLONING DATASET MODE: OFF")


def toggle_dataset():
    set_dataset_mode(not dataset_mode)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def try_capture(
    frame,
    angle: float,
    speed: float,
    brake: float,
    current_time: float,
    autonomous_mode: bool = False,
    line_detected: bool = False,
    gps_vals=None,
    lidar_obstacle_detected: bool = False,
    lidar_obstacle_distance=None,
    evasion_phase: int = 0,
    obstacle_type: str = "",
):
    """
    Attempt to capture one dataset sample if enough time has passed.

    Parameters
    ----------
    frame                       BGR image from Webots camera.
    angle                       Steering angle in radians (applied this frame).
    speed                       Cruising speed in km/h (applied this frame).
    brake                       Brake intensity 0–1 (applied this frame).
    current_time                time.time() value.
    autonomous_mode             True when PID is driving, False for manual.
    line_detected               Whether yellow lane line was found this frame.
    gps_vals                    GPS (x, y, z) in metres, or None.
    lidar_obstacle_detected     True when LiDAR sees something in the front arc.
    lidar_obstacle_distance     Closest LiDAR reading in metres, or None.
    evasion_phase               0=NORMAL, 1=EVADING, 2=RETURNING.
    obstacle_type               Camera-confirmed type: VEHICLE, OBSTACLE, LIDAR_ONLY, etc.

    Returns
    -------
    (counter, overlay) — counter is int; overlay is always None.
    """
    global dataset_counter, last_capture_time

    if not dataset_mode:
        return dataset_counter, None

    if (current_time - last_capture_time) < CAPTURE_INTERVAL:
        return dataset_counter, None

    # ── Capture ──────────────────────────────────────────────
    dataset_counter += 1
    last_capture_time = current_time

    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(current_time))
    timestamp = ts + f".{int((current_time % 1) * 1000):03d}"

    filename = f"frame_{dataset_counter:06d}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)

    capture_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    cv2.imwrite(filepath, capture_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    if _csv_writer:
        gps_x = f"{gps_vals[0]:.3f}" if gps_vals is not None else ""
        gps_y = f"{gps_vals[1]:.3f}" if gps_vals is not None else ""
        lidar_dist_str = f"{lidar_obstacle_distance:.2f}" if lidar_obstacle_distance is not None else ""
        _csv_writer.writerow([
            _session_id,
            timestamp,
            filename,
            f"{angle:.4f}",
            f"{speed:.2f}",
            f"{brake:.4f}",
            int(autonomous_mode),
            int(line_detected),
            gps_x,
            gps_y,
            int(lidar_obstacle_detected),
            lidar_dist_str,
            evasion_phase,
            obstacle_type,
        ])
        _csv_file_handle.flush()

    if dataset_counter % 10 == 0:
        print(f"[DS] Sample #{dataset_counter} | angle={angle:.3f} "
              f"speed={speed:.1f} brake={brake:.2f} "
              f"auto={int(autonomous_mode)} line={int(line_detected)} "
              f"phase={evasion_phase} type={obstacle_type or '--'}")

    return dataset_counter, None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def get_status():
    return {
        "mode": dataset_mode,
        "samples": dataset_counter,
        "session": _session_id,
        "path": IMAGES_DIR,
        "csv": CSV_PATH,
    }


# ---------------------------------------------------------------------------
# Module-level init
# ---------------------------------------------------------------------------
init_dataset()
