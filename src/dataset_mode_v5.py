# ============================================================
# dataset_mode_v5 — CIL (Conditional Imitation Learning) Dataset Module
# ============================================================
#
# PURPOSE
#   Capture synchronized camera frames + control labels for training
#   a branched CIL model that responds to explicit navigation commands
#   (LANE_FOLLOW, STRAIGHT, LEFT, RIGHT) at intersections.
#
# WHAT IS NEW vs v4
# ──────────────────
#   1. nav_command column — the CIL training label.
#      Set by the controller via set_nav_command() before each intersection.
#      Auto-resets to LANE_FOLLOW when edge/line is reacquired after the turn.
#
#   2. lane_side column — metadata indicating which lane the car was in.
#      Used in the training notebook to filter or split datasets per lane.
#      NOT a model input — the visual difference between lanes is strong
#      enough for the model to learn lane context from the image itself.
#
#   3. edge_detected column — whether the gravel/pavement edge was visible
#      this frame. Right-lane analog to line_detected.
#      Detected by the controller (detect_gravel_edge) and passed in,
#      exactly the same pattern as line_detected.
#
#   4. Dynamic capture rate:
#      - 2 Hz (0.50 s interval) during normal lane following
#      - 4 Hz (0.25 s interval) when nav_command != LANE_FOLLOW
#        (approaching or inside an intersection — timing is critical)
#
# NAV COMMAND VALUES
# ───────────────────
#   0  LANE_FOLLOW   default — car is following the lane, no intersection intent
#   1  STRAIGHT      driver intends to cross the next intersection straight
#   2  LEFT          driver intends to turn left at the next intersection
#   3  RIGHT         driver intends to turn right at the next intersection
#
# NAV COMMAND LIFECYCLE (set from controller)
# ─────────────────────────────────────────────
#   1. Driver approaches intersection, presses PS4 button to set command:
#        L1  →  set_nav_command(LEFT)
#        R1  →  set_nav_command(RIGHT)
#        O   →  set_nav_command(STRAIGHT)
#   2. Command latches — stays active through the approach AND the full turn.
#      These are the most important frames for CIL training.
#   3. After intersection, edge/line reacquired → controller calls reset_nav_command().
#      Module auto-resets to LANE_FOLLOW.
#
# LANE SIDE VALUES
# ─────────────────
#   0  LEFT_LANE    car is following the yellow centerline (left lane)
#   1  RIGHT_LANE   car is following the gravel edge (right lane)
#
# MULTI-CONTRIBUTOR SETUP
#   Same as v4. Each team member sets a unique CONTRIBUTOR_ID:
#
#   Option A — edit this file (change the default "A01" below).
#   Option B — set environment variable before running Webots:
#       Windows:  $env:DATASET_CONTRIBUTOR_ID = "B02"
#       Linux/Mac: export DATASET_CONTRIBUTOR_ID=B02
#
#   Agreed IDs for this team:
#       A01 — Angel Barajas
#       B02 — <partner 2 name>
#       C03 — <partner 3 name>
#
# CSV COLUMNS
#   contributor_id          — short ID of the person who recorded this row
#   session_id              — recording session identifier (YYYYmmdd_HHMMSS)
#   timestamp               — ISO-8601 with milliseconds (UTC)
#   image_filename          — {ID}_frame_XXXXXX.jpg
#   steering_angle          — radians (applied this frame)
#   speed_kmh               — km/h (applied this frame)
#   brake                   — 0–1 (applied this frame)
#   autonomous_mode         — 0=manual (expert), 1=autonomous
#   lane_side               — 0=LEFT_LANE, 1=RIGHT_LANE
#   line_detected           — 0/1 yellow centerline visible (left lane reference)
#   edge_detected           — 0/1 gravel/pavement edge visible (right lane reference)
#   gps_x, gps_y            — metres (blank if GPS unavailable)
#   lidar_obstacle_detected — 0/1 LiDAR sees object in 40° front arc
#   lidar_obstacle_distance — metres to closest object (blank if none)
#   behavior_class          — 0=NORMAL, 1=EVADING, 2=RETURNING,
#                             3=SHARP_TURN_LEFT, 4=SHARP_TURN_RIGHT
#   nav_command             — 0=LANE_FOLLOW, 1=STRAIGHT, 2=LEFT, 3=RIGHT
#   obstacle_type           — VEHICLE | OBSTACLE | LIDAR_ONLY |
#                             OBJECT_AHEAD | "" (empty = clear)
#
# OUTPUT
#   data/behavioral_dataset_CIL_06242026/images/{ID}_frame_XXXXXX.jpg
#   data/behavioral_dataset_CIL_06242026/measurements.csv  (append mode)
#
# CAPTURE RATE
#   2 Hz during LANE_FOLLOW, 4 Hz when nav_command is active.
#
# USAGE
#   import dataset_mode_v5 as dataset_mode
#
#   # Set nav command from PS4 button handler:
#   dataset_mode.set_nav_command(dataset_mode.CMD_LEFT)
#
#   # After intersection clears:
#   dataset_mode.reset_nav_command()
#
#   # Capture frame (call every loop iteration):
#   dataset_mode.try_capture(
#       frame, angle, speed, brake, current_time,
#       autonomous_mode=autonomous_mode,
#       lane_side=lane_side,
#       line_detected=line_detected,
#       edge_detected=edge_detected,
#       gps_vals=gps_vals,
#       lidar_obstacle_detected=lidar_obstacle,
#       lidar_obstacle_distance=lidar_obstacle_distance,
#       behavior_class=behavior_class,
#       obstacle_type=obstacle_type,
#   )
# ============================================================

import os
import csv
import time
import cv2

# ──────────────────────────────────────────────────────────────
# Nav command constants — import these in the controller so
# there are no magic numbers in either file.
# ──────────────────────────────────────────────────────────────
CMD_LANE_FOLLOW = 0
CMD_STRAIGHT    = 1
CMD_LEFT        = 2
CMD_RIGHT       = 3

_CMD_NAMES = {
    CMD_LANE_FOLLOW: "LANE_FOLLOW",
    CMD_STRAIGHT:    "STRAIGHT",
    CMD_LEFT:        "LEFT",
    CMD_RIGHT:       "RIGHT",
}

# Lane side constants
LANE_LEFT  = 0
LANE_RIGHT = 1

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
CAPTURE_INTERVAL_NORMAL       = 0.50   # 2 Hz — normal lane following
CAPTURE_INTERVAL_INTERSECTION = 0.25   # 4 Hz — nav_command active (approach + turn)

JPEG_QUALITY = 85
IMG_WIDTH    = 620
IMG_HEIGHT   = 320

# Each contributor sets their own short ID (see header above).
CONTRIBUTOR_ID = os.environ.get("DATASET_CONTRIBUTOR_ID", "A01")

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "behavioral_dataset_CIL_06252026"
)
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH   = os.path.join(DATASET_DIR, "measurements.csv")

CSV_HEADER = [
    "contributor_id", "session_id", "timestamp", "image_filename",
    "steering_angle", "speed_kmh", "brake",
    "autonomous_mode",
    "lane_side",
    "line_detected",
    "edge_detected",
    "gps_x", "gps_y",
    "lidar_obstacle_detected", "lidar_obstacle_distance",
    "behavior_class",
    "nav_command",
    "obstacle_type",
]

# ──────────────────────────────────────────────────────────────
# Module state
# ──────────────────────────────────────────────────────────────
dataset_mode      = False
dataset_counter   = 0
last_capture_time = 0.0
_session_id       = ""
_csv_file_handle  = None
_csv_writer       = None
_nav_command      = CMD_LANE_FOLLOW   # current navigation intent, set by controller


# ──────────────────────────────────────────────────────────────
# Nav command API — called from the controller
# ──────────────────────────────────────────────────────────────

def set_nav_command(cmd: int):
    """
    Set the active navigation command. Call this from the PS4/keyboard handler
    when the driver presses L1 (LEFT), R1 (RIGHT), or O (STRAIGHT).
    The command latches until reset_nav_command() is called.
    """
    global _nav_command
    _nav_command = cmd
    print(f"[DS] nav_command → {_CMD_NAMES.get(cmd, '?')}  "
          f"(capture rate: {'4 Hz' if cmd != CMD_LANE_FOLLOW else '2 Hz'})")


def get_nav_command() -> int:
    """Return the current nav command. Controller reads this each frame."""
    return _nav_command


def reset_nav_command():
    """
    Reset nav_command to LANE_FOLLOW. Call this from the controller when
    edge_detected or line_detected becomes True after an intersection,
    indicating the car has successfully completed the commanded turn.
    """
    global _nav_command
    if _nav_command != CMD_LANE_FOLLOW:
        print(f"[DS] nav_command reset → LANE_FOLLOW  "
              f"(was {_CMD_NAMES.get(_nav_command, '?')})")
    _nav_command = CMD_LANE_FOLLOW


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _count_existing_frames() -> int:
    """Count this contributor's frames on disk so the counter never resets."""
    if not os.path.isdir(IMAGES_DIR):
        return 0
    prefix = f"{CONTRIBUTOR_ID}_frame_"
    return sum(1 for f in os.listdir(IMAGES_DIR)
               if f.startswith(prefix) and f.endswith(".jpg"))


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
        print(f"[DS] Created new measurements.csv  ({CSV_PATH})")
    else:
        _csv_file_handle = open(CSV_PATH, "a", newline="")
        _csv_writer = csv.writer(_csv_file_handle)
        print(f"[DS] Appending to existing measurements.csv  ({CSV_PATH})")


def _close_csv():
    global _csv_file_handle, _csv_writer
    if _csv_file_handle and not _csv_file_handle.closed:
        _csv_file_handle.close()
        print(f"[DS] measurements.csv saved.  Total samples this session: {dataset_counter}")


def init_dataset():
    """Create dataset directories. Called automatically at module import."""
    _ensure_dirs()


# ──────────────────────────────────────────────────────────────
# Toggle
# ──────────────────────────────────────────────────────────────

def set_dataset_mode(on: bool):
    global dataset_mode, dataset_counter, last_capture_time, _session_id
    dataset_mode = on
    if on:
        dataset_counter   = _count_existing_frames()
        last_capture_time = 0.0
        _session_id       = time.strftime("%Y%m%d_%H%M%S")
        _open_csv()
        print(f"[DS] CIL DATASET MODE: ON  |  contributor={CONTRIBUTOR_ID}  "
              f"session={_session_id}  resuming at frame #{dataset_counter}")
        print(f"[DS] Rate: 2 Hz (LANE_FOLLOW)  /  4 Hz (nav_command active)")
        print(f"[DS] PS4: L1=LEFT  R1=RIGHT  O=STRAIGHT  (set before intersection)")
    else:
        _close_csv()
        print("[DS] CIL DATASET MODE: OFF")


def toggle_dataset():
    set_dataset_mode(not dataset_mode)


# ──────────────────────────────────────────────────────────────
# Capture
# ──────────────────────────────────────────────────────────────

def try_capture(
    frame,
    angle:   float,
    speed:   float,
    brake:   float,
    current_time: float,
    autonomous_mode:          bool  = False,
    lane_side:                int   = LANE_LEFT,
    line_detected:            bool  = False,
    edge_detected:            bool  = False,
    gps_vals                        = None,
    lidar_obstacle_detected:  bool  = False,
    lidar_obstacle_distance         = None,
    behavior_class:           int   = 0,
    obstacle_type:            str   = "",
):
    """
    Attempt to capture one CIL dataset sample if enough time has passed.

    Capture rate is 2 Hz during normal lane following and 4 Hz when a
    nav_command is active (approaching or inside an intersection).
    The nav_command value written to CSV is always read from the module-level
    _nav_command state — set it with set_nav_command() before calling this.

    Parameters
    ----------
    frame                       BGR image from Webots camera (full resolution).
    angle                       Steering angle in radians (applied this frame).
    speed                       Cruising speed in km/h (applied this frame).
    brake                       Brake intensity 0–1 (applied this frame).
    current_time                time.time() value.
    autonomous_mode             True when autonomous mode is active.
    lane_side                   LANE_LEFT (0) or LANE_RIGHT (1).
    line_detected               Yellow centerline visible this frame.
    edge_detected               Gravel/pavement edge visible this frame.
    gps_vals                    GPS (x, y, z) in metres, or None.
    lidar_obstacle_detected     True when LiDAR sees something in the front arc.
    lidar_obstacle_distance     Closest LiDAR reading in metres, or None.
    behavior_class              0=NORMAL … 4=SHARP_RIGHT (see dataset_mode_v4).
    obstacle_type               Camera-confirmed type string or "".

    Returns
    -------
    (counter, overlay) — counter is int; overlay is always None.
    """
    global dataset_counter, last_capture_time

    if not dataset_mode:
        return dataset_counter, None

    # Dynamic interval — 4 Hz when a nav command is active
    interval = (CAPTURE_INTERVAL_INTERSECTION
                if _nav_command != CMD_LANE_FOLLOW
                else CAPTURE_INTERVAL_NORMAL)

    if (current_time - last_capture_time) < interval:
        return dataset_counter, None

    # ── Capture ──────────────────────────────────────────────
    dataset_counter  += 1
    last_capture_time = current_time

    ts        = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(current_time))
    timestamp = ts + f".{int((current_time % 1) * 1000):03d}"

    filename = f"{CONTRIBUTOR_ID}_frame_{dataset_counter:06d}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)

    capture_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    cv2.imwrite(filepath, capture_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    if _csv_writer:
        gps_x        = f"{gps_vals[0]:.3f}" if gps_vals is not None else ""
        gps_y        = f"{gps_vals[1]:.3f}" if gps_vals is not None else ""
        lidar_dist   = (f"{lidar_obstacle_distance:.2f}"
                        if lidar_obstacle_distance is not None else "")
        _csv_writer.writerow([
            CONTRIBUTOR_ID,
            _session_id,
            timestamp,
            filename,
            f"{angle:.4f}",
            f"{speed:.2f}",
            f"{brake:.4f}",
            int(autonomous_mode),
            lane_side,
            int(line_detected),
            int(edge_detected),
            gps_x,
            gps_y,
            int(lidar_obstacle_detected),
            lidar_dist,
            behavior_class,
            _nav_command,
            obstacle_type,
        ])
        _csv_file_handle.flush()

    # Progress log every 10 samples
    if dataset_counter % 10 == 0:
        _BC_CLASS  = {0: "NORMAL", 1: "EVADING", 2: "RETURNING",
                      3: "SHARP_LEFT", 4: "SHARP_RIGHT"}
        _LANE_NAME = {LANE_LEFT: "LEFT", LANE_RIGHT: "RIGHT"}
        print(
            f"[DS] #{dataset_counter:06d}  "
            f"cmd={_CMD_NAMES.get(_nav_command,'?'):11s}  "
            f"lane={_LANE_NAME.get(lane_side,'?'):5s}  "
            f"angle={angle:.3f}  speed={speed:.1f}  "
            f"line={int(line_detected)}  edge={int(edge_detected)}  "
            f"class={behavior_class}({_BC_CLASS.get(behavior_class,'?')})"
        )

    return dataset_counter, None


# ──────────────────────────────────────────────────────────────
# Status
# ──────────────────────────────────────────────────────────────

def get_status() -> dict:
    return {
        "mode":        dataset_mode,
        "samples":     dataset_counter,
        "session":     _session_id,
        "nav_command": _CMD_NAMES.get(_nav_command, "?"),
        "path":        IMAGES_DIR,
        "csv":         CSV_PATH,
    }


# ──────────────────────────────────────────────────────────────
# Module-level init
# ──────────────────────────────────────────────────────────────
init_dataset()
