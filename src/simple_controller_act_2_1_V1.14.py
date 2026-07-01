# ============================================================
# V1.14 — Webots Self-Driving Controller
# ============================================================
#
# WHAT IS NEW IN V1.14 vs V1.13
# ──────────────────────────────
#   1. CIL (Conditional Imitation Learning) inference integrated into autonomous mode.
#      Model: models/cil_model.pt — trained in Notebook 28 (BehavioralCloning_CIL.ipynb).
#      Architecture: shared DAVE-2 conv trunk + 4 FC branches, one per nav_command.
#      Reference: Codevilla et al. (2018) — End-to-end Driving via Conditional Imitation
#      Learning. Human provides high-level intent (L1/R1/Circle); CIL executes steering.
#
#   2. CIL arbitration — slot inserted inside the normal-driving block
#      (evasion_phase=0, no LiDAR emergency), before BC/PID modes A–D:
#
#        nav_cmd == LANE_FOLLOW (0) → BC/PID arbitration unchanged (V1.13 behaviour)
#        nav_cmd == STRAIGHT   (1) → CIL steers straight through intersection
#        nav_cmd == LEFT       (2) → CIL steers left  through intersection
#        nav_cmd == RIGHT      (3) → CIL steers right through intersection
#
#      While CIL is active: speed = CIL_SPEED (15 km/h), brake = 0.
#      EMA smoothing (α=0.40) and hard clamp (±0.70 rad) identical to BC pipeline.
#      CIL does NOT activate during evasion phases or LiDAR stop/slow zones —
#      safety overrides always take priority.
#
#   3. nav_command lifecycle in autonomous mode:
#        a. Co-pilot presses PS4 L1 / R1 / Circle before the intersection.
#        b. nav_cmd latches (CMD_LEFT / CMD_RIGHT / CMD_STRAIGHT).
#        c. CIL takes over steering at CIL_SPEED; BC/PID are bypassed.
#        d. Car exits intersection → gravel edge or yellow line reacquired.
#        e. 3-second hold timer starts (NAV_CMD_RESET_DELAY = 3.0 s).
#        f. Timer expires → nav_cmd resets to CMD_LANE_FOLLOW → BC/PID resumes.
#
#   4. Fallback when cil_model.pt not found:
#      nav_cmd ≠ LANE_FOLLOW falls back to existing BC/PID arbitration (V1.13
#      behaviour) with a one-time warning printed at startup. Controller is
#      fully operational without the CIL model — it just won't use CIL inference.
#
#   5. CIL preprocessing — must match CILDataset._load_and_crop (Notebook 28) exactly:
#        - Crop: remove top 40 % (sky), keep until 95 % height (removes hood).
#        - Convert BGR → RGB.
#        - Resize to 320 × 160 px.
#        - Normalize to float32: (pixel/255 − MEAN) / STD  (ImageNet statistics).
#        - Output tensor: (1, 3, 160, 320) on _CIL_DEVICE (CUDA if available).
#
#   6. ADAS Monitor additions (panel height 948 → 971):
#        - New "CIL model" row: NOT FOUND / LOADED (standby) / ACTIVE (command).
#        - Control row: shows "CIL  LEFT / RIGHT / STRAIGHT" in gold when CIL active.
#        - Nav cmd row: command label turns gold while CIL is executing it.
#
#   7. All V1.13 features retained unchanged:
#      gravel edge detection, BC/PID arbitration, nav_command latch+reset (3 s),
#      evasion phases, lane_side toggle, dataset_mode_v5 capture.
#
# WHAT IS NEW IN V1.13 vs V1.12
# ──────────────────────────────
#   1. Gravel/pavement edge detector (detect_gravel_edge) for right-lane keeping.
#      Returns edge_detected (bool), edge_x (px), edge_error (px from target).
#   2. Right-lane PID mode: PID uses edge_error from gravel edge instead of
#      yellow-line error when lane_side = LANE_RIGHT.
#   3. lane_side toggle (key N): LEFT_LANE (yellow line) ↔ RIGHT_LANE (gravel edge).
#      Resets PID integrator on each toggle to prevent error carryover.
#   4. Navigation commands via PS4 L1/R1/Circle latch until intersection is cleared,
#      then hold for NAV_CMD_RESET_DELAY (3.0 s) before auto-resetting to LANE_FOLLOW.
#   5. Dataset capture upgraded to dataset_mode_v5:
#        - New CSV columns: lane_side, edge_detected, nav_command.
#        - Capture rate: 4 Hz when nav_command ≠ LANE_FOLLOW, 2 Hz otherwise.
#        - New dataset directory: behavioral_dataset_CIL_<date>.
#   6. ADAS Monitor extended: lane_side, edge_detected, nav_command rows added.
#
# CONTROLLER PRIORITY ORDER (autonomous mode, highest to lowest)
# ──────────────────────────────────────────────────────────────
#   1. Evasion phase 1 (EVADING)  — vehicle alongside: BC or DS_ML parallel control
#   2. Evasion phase 2 (RETURNING) — vehicle cleared:  DS_MR-guided return to lane
#   3. LiDAR emergency stop        — obstacle < 10 m:  full brake, angle = 0
#   4. LiDAR slow zone             — obstacle 10–25 m: PID at 50 % speed
#   5. CIL mode                    — nav_cmd ≠ LANE_FOLLOW + model loaded
#   6. BC/PID arbitration (A–D)    — nav_cmd = LANE_FOLLOW or CIL model missing
#
# CIL MODEL PIPELINE (per frame, when nav_cmd ≠ LANE_FOLLOW)
# ────────────────────────────────────────────────────────────
#   1. Crop frame: remove top CIL_CROP_TOP (40 %) and keep until CIL_CROP_BOT (95 %).
#   2. Convert BGR → RGB.
#   3. Resize to CIL_IMG_W × CIL_IMG_H (320 × 160 px).
#   4. Normalize: float32 pixel/255, then subtract ImageNet mean, divide by std.
#   5. Run TorchScript inference via model(tensor, nav_command) on _CIL_DEVICE.
#   6. Clamp raw output to ± CIL_ANGLE_CLAMP (0.70 rad).
#   7. EMA smooth: smoothed = CIL_SMOOTH_ALPHA × raw + (1−α) × prev.
#      α=0.40 → ~80 ms lag at 32 ms timestep. EMA state resets to 0 on deactivation.
#
# BC / PID ARBITRATION — four operating modes (active when nav_cmd = LANE_FOLLOW)
# ─────────────────────────────────────────────────────────────────────────────────
#   B=ON  L=ON   HYBRID    — PID handles behavior class 0 (line visible, normal road);
#                             BC handles classes 1–4 (evasion, returning, sharp turns).
#                             Best of both: PID accuracy on straight + BC for maneuvers.
#
#   B=ON  L=OFF  FULL BC   — BC controls all classes including class 0.
#                             Use to validate BC on full circuit without PID assistance.
#
#   B=OFF L=ON   PURE PID  — BC model loaded but never called. Identical to V1.11.
#                             Safe baseline; line-lost fallback active (10 s countdown).
#
#   B=OFF L=OFF  SAFE HOLD — Both controllers disabled. Angle=0, speed=0, brake=0.3.
#                             Emergency fallback — press S to exit autonomous mode.
#
# BC MODEL PIPELINE (per frame, when BC is active)
# ──────────────────────────────────────────────────
#   1. Crop frame: remove top BC_CROP_TOP (38 %) and keep until BC_CROP_BOT (88 %).
#   2. Resize to BC_IMG_W × BC_IMG_H (200 × 66 px) — NVIDIA DAVE-2 input size.
#   3. Normalize to [0.0, 1.0] float32 (no mean/std shift — matches N27 training).
#   4. Run TorchScript inference on _BC_DEVICE (CUDA if available, otherwise CPU).
#   5. Clamp raw output to ± BC_ANGLE_CLAMP (0.70 rad).
#   6. EMA smooth: smoothed = BC_SMOOTH_ALPHA × raw + (1−α) × prev.
#      α=0.40 → ~80 ms lag. Same convention as CIL.
#
# BC SPEED BY CLASS
# ──────────────────
#   Speed set per behavior_class while BC is active. Edit BC_SPEED_BY_CLASS below.
#   Class 0 NORMAL          30 km/h — steady lane following
#   Class 1 EVADING         30 km/h — maintain momentum while passing
#   Class 2 RETURNING       30 km/h — merge back at comfortable speed
#   Class 3 SHARP_TURN_LEFT 15 km/h — slow for precise left intersection
#   Class 4 SHARP_TURN_RIGHT15 km/h — slow for precise right intersection
#
# BC STUCK DETECTION
# ───────────────────
#   While BC is commanding movement (speed > BC_STUCK_SPEED_THRESHOLD) and the
#   actual vehicle speed (driver.getCurrentSpeed()) stays below the threshold for
#   more than BC_STUCK_TIME_THRESHOLD (5 s), a WARNING is printed to the console.
#   Timer resets after each warning so the alert repeats every 5 s.
#   Action: press S to exit autonomous mode and manually unstick the vehicle.
#
# LANE FOLLOWING  (PID)
# ──────────────────────
#   LEFT LANE (lane_side = LANE_LEFT — yellow line reference):
#     - Grayscale brightness threshold (120–200) masks the yellow centerline.
#     - Narrow trapezoidal ROI + morphological open/close/dilate before Canny+Hough.
#     - Weighted-average Hough line fit with slope (> MAX_SLOPE) and boundary guards.
#     - Moving-average smoothing (deque of 10 frames) + exponential temporal filter.
#     - Virtual lane center = yellow-line X + LANE_OFFSET_PX (110 px rightward).
#     - PID error = lane_center_x − camera_center_x.
#     - Line-lost fallback (PID-only mode): hold last angle × 0.5 decay at 10 km/h
#       for 10 s, then revert to MANUAL automatically.
#
#   RIGHT LANE (lane_side = LANE_RIGHT — gravel edge reference):
#     - HSV masking + CLAHE V-channel equalization isolates gravel/sandy surface.
#     - Leftmost gravel column = road edge (pavement-to-gravel boundary).
#     - PID error = edge_x − target_x  (target = EDGE_TARGET_X_RATIO × frame width).
#     - Positive error → edge right of target → car drifted left → steer right.
#     - Tune GRAVEL_H/S/V_MIN/MAX in config section for specific Webots world.
#
#   Common PID settings (both lanes):
#     KP=0.0015, KI=0.000, KD=0.0003. Output clamped to ±STEERING_GAIN_LIMIT.
#     Steering smoothed per frame: 60 % previous + 40 % PID output.
#     Speed reduced proportionally with |error|; floor at 8 km/h.
#
# GRAVEL EDGE DETECTION CONFIG
# ─────────────────────────────
#   EDGE_TARGET_X_RATIO = 0.82  — desired edge position as fraction of frame width.
#   EDGE_CROP_TOP / BOT  = 0.70 / 0.95 — ROI rows in the full frame.
#   GRAVEL_H/S/V_MIN/MAX — HSV color range for the gravel surface. Tune in Webots:
#     1. Press N (RIGHT_LANE) + P (debug panel). Pause on a straight section.
#     2. Sample gravel and asphalt pixels with an HSV color picker.
#     3. Set H/S/V bounds to cover gravel, exclude asphalt.
#     4. Verify Gravel Edge Debug window: gravel=white, road=black.
#   CLAHE clipLimit=2.0, tileGridSize=(8,8) normalizes brightness locally so
#   shadowed gravel stays detectable alongside lit gravel.
#
# LIDAR OBSTACLE DETECTION  (Sick LMS 291)
# ──────────────────────────────────────────
#   - Front-center 40° inspection window (±20 samples from center of 180 samples).
#   - Max detection range LIDAR_MAX_DISTANCE: 25 m.
#   - Slow zone (10–25 m): speed = 50 % AUTONOMOUS_SPEED; steering unchanged.
#   - Emergency stop (< LIDAR_STOP_DISTANCE = 10 m): full brake, angle = 0.
#   - LiDAR safety always takes priority over CIL, BC, and PID.
#
# CAMERA RECOGNITION  (Webots built-in neural network)
# ──────────────────────────────────────────────────────
#   - camera.recognitionEnable() — no external training required.
#   - Vehicles detected: bus, car, truck, van, toyota, lincoln, citroen, bmw,
#     mercedes, suv. Own car (bmw model) is auto-skipped.
#   - Also recognises: traffic_light, sign (caution/order/speed/yield/highway),
#     obstacle (traffic cone, barrel, crash barrier). Buildings/trees ignored.
#   - Color-coded bounding boxes drawn in debug view:
#       red = vehicle, yellow = traffic_light, cyan = sign, blue = obstacle.
#   - Classification runs every frame in all modes for dataset labeling.
#   - Only objects confirmed within LIDAR_STOP_DISTANCE (10 m) trigger evasion.
#
# OBSTACLE CLASSIFICATION & EVASION PHASES
# ──────────────────────────────────────────
#   obstacle_type values: VEHICLE | OBSTACLE | LIDAR_ONLY | OBJECT_AHEAD | ""
#   vehicle_nearby: True when a vehicle is confirmed < 10 m by BOTH LiDAR + camera.
#   Evasion triggers ONLY for vehicles (hazard lights ON + evasion state machine).
#   Non-vehicle obstacles (cones, barriers) → full brake only, no evasion.
#
#   Phase | Name      | Trigger / Exit
#   ------+-----------+--------------------------------------------------
#     0   | NORMAL    | default; entered from RETURNING when line/wZ/timeout
#     1   | EVADING   | vehicle_nearby=True; exits when left DS clear + latch
#     2   | RETURNING | left sensors cleared; exits on line/wZ align/timeout
#
#   Phase 1 sub-phases:
#     A (_left_ds_triggered=False): BC steers right past the obstacle.
#     B (_left_ds_triggered=True ): DS_ML proportional control holds lateral gap.
#   Phase 2: DS_MR keeps distance from right guard; base left steer merges back.
#   Saved gyro wZ at phase 1 entry used as heading reference for phase 2 exit.
#
# BEHAVIOR CLASS — dataset label, computed every frame
# ──────────────────────────────────────────────────────
#   Auto-detected from sensor+angle state; no manual annotation needed.
#   Priority: evasion (1/2) > sharp turn (3/4) > normal (0).
#   Sharp turn debounce: SHARP_TURN_ENTRY_ANGLE=0.25 rad + SHARP_TURN_DWELL_SECS=4 s.
#
#   Class | Name            | Trigger
#   ------+-----------------+-----------------------------------------
#     0   | NORMAL          | default state
#     1   | EVADING         | vehicle_nearby AND LiDAR < 10 m
#     2   | RETURNING       | just exited EVADING; DS sensors clear
#     3   | SHARP_TURN_LEFT | angle < −0.25 rad, no line, no evasion
#     4   | SHARP_TURN_RIGHT| angle > +0.25 rad, no line, no evasion
#
# SENSORS  (GPS + Gyro + 6× Distance Sensors)
# ─────────────────────────────────────────────
#   Devices: gps, gyro, DS_ML, DS_MR, DS_FL, DS_FR, DS_RL, DS_RR.
#   Each enabled at startup; skipped gracefully if not found in Webots world.
#   Values read every frame and forwarded to ADAS Monitor and dataset capture.
#   DS_ML/DS_FL/DS_RL used for evasion latch detection (left side clearance).
#   DS_MR used for RETURNING phase lateral correction (right guard distance).
#
# ADAS MONITOR  (draw_adas_monitor)
# ────────────────────────────────────
#   Separate OpenCV window toggled with key A. Panel size: 400 × 971 px (V1.14).
#   Sections and rows:
#     MACHINE STATE: Mode, Phase, Control, BC model, PID/line, CIL model (NEW V1.14),
#                    Lane side, Gravel edge, Nav cmd, Yellow line, wZ align (RETURNING only)
#     VEHICLE STATE: Speed (km/h), Angle (rad), Brake
#     GPS          : X, Y, Z in metres
#     GYRO         : wX, wY, wZ in rad/s
#     DISTANCE     : DS_ML, DS_MR, DS_FL, DS_FR, DS_RL, DS_RR (color: green/orange/red)
#     LIDAR        : Status, Distance (m), Angle (deg), Type, Vehicle confirmed
#   Color conventions: green = nominal, orange = caution, red = critical, gold = CIL active.
#
# DATASET CAPTURE  (dataset_mode_v5)
# ────────────────────────────────────
#   Toggle: keyboard D or PS4 Triangle.
#   Capture rate: 4 Hz when nav_command ≠ LANE_FOLLOW, 2 Hz otherwise.
#   Capture point: after all control decisions — angle/speed/brake reflect the
#   actual command applied this frame (CIL, BC, or PID — transparent to dataset).
#   Works in both manual and autonomous mode.
#   Multi-contributor: set CONTRIBUTOR_ID in dataset_mode_v5.py or via
#     $env:DATASET_CONTRIBUTOR_ID before launching Webots.
#   CSV columns: contributor_id, session_id, timestamp, image_filename,
#     steering_angle, speed_kmh, brake, autonomous_mode, lane_side,
#     line_detected, edge_detected, gps_x, gps_y, lidar_obstacle_detected,
#     lidar_obstacle_distance, behavior_class, nav_command, obstacle_type.
#   Images → data/behavioral_dataset_CIL_<date>/images/{ID}_frame_XXXXXX.jpg
#   Metadata → data/behavioral_dataset_CIL_<date>/measurements.csv (append mode)
#   Frame counter resumes from existing files — toggling D never overwrites data.
#
# INPUT / CONTROL
# ─────────────────
#   Keyboard:
#     Arrow keys  — manual steering (Left/Right) and speed (Up/Down)
#     S           — toggle autonomous mode
#     A           — toggle ADAS Monitor window
#     B           — toggle BC inference (ON / OFF)
#     L           — toggle PID / line-follower (ON / OFF)
#     N           — toggle lane side (LEFT_LANE ↔ RIGHT_LANE)
#     C           — save single camera frame to file
#     D           — toggle dataset capture
#     P           — toggle debug panel (Self Driving Debug + Mask windows)
#     Q           — quit
#   PS4 controller:
#     Left stick  — steering (manual mode only)
#     R2          — throttle
#     L2          — brake / reverse
#     X button    — toggle autonomous mode
#     Square      — save camera frame
#     Triangle    — toggle dataset capture
#     L1          — nav_command = CMD_LEFT    (turn left at next intersection)
#     R1          — nav_command = CMD_RIGHT   (turn right at next intersection)
#     Circle (O)  — nav_command = CMD_STRAIGHT (go straight at next intersection)
#   Debounce: 0.1 s keyboard, 0.5 s PS4 buttons.
#
# DEBUG DISPLAYS
# ───────────────
#   "Self Driving Debug"   — annotated camera view: lane reference lines, HUD text
#                            (mode, angle, obstacle, behavior class), dataset indicator.
#                            CIL active: gold text "AUTO | CIL LEFT/RIGHT/STRAIGHT angle:±x.xxx".
#   "Yellow Mask Debug"    — binary brightness mask (LEFT lane mode, key N).
#   "Gravel Edge Debug"    — binary HSV gravel mask (RIGHT lane mode, key N).
#                            Swaps automatically with lane_side; old window destroyed.
#   "PID Response Chart"   — live traces: error (px, cyan), steering (rad, green),
#                            speed (km/h, magenta). Each signal has its own zero line.
#   "ADAS Monitor"         — full sensor dashboard (see ADAS MONITOR section above).
#   Speed overlay          — bottom-left of debug view: cyan = autonomous, green = manual.
#   DEBUG_PANEL (key P)    — gates Self Driving Debug + Mask windows together.
#   ADAS_MONITOR (key A)   — gates ADAS Monitor independently.
# ============================================================

from controller import Display, Keyboard, Lidar
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time
import pygame
from collections import deque
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_mode_v5 as dataset_mode

# ==============================
# BEHAVIORAL CLONING — imports
# ==============================
try:
    import torch as _torch
    _BC_TORCH_AVAILABLE = True
except ImportError:
    _BC_TORCH_AVAILABLE = False
    print("[BC] WARNING: torch not installed — BC model disabled.")


# ==============================
# MOVING AVERAGE LINEX FROM PREVIOUS FRAMES
# ==============================
line_x_history = deque(maxlen=10)


# ==============================
# CONFIGURATION CONSTANTS
# ==============================

DEBOUNCE_TIME = 0.1

MAX_ANGLE = 0.7  # before 0.5
MAX_SPEED = 100

SPEED_INCR = 5
ANGLE_INCR = 0.05

# PS4 controller tuning
DEADZONE = 0.10
CONTROLLER_MAX_SPEED = 80
REVERSE_MAX_SPEED = -20

# Autonomous driving tuning
AUTONOMOUS_SPEED = 50
STEERING_GAIN_LIMIT = 0.7 #before 0.5

# PID tuning
KP = 0.0015 #before 0.0013
KI = 0.000
KD = 0.0003

# Yellow line smoothing
yellow_line_prev = None
alpha_smooth = 0.25

# PS4 button map
BTN_X        = 0   # Cross    — toggle autonomous mode
BTN_CIRCLE   = 1   # Circle   — nav_command STRAIGHT
BTN_TRIANGLE = 2   # Triangle — toggle dataset mode
BTN_SQUARE   = 3   # Square   — save image
BTN_L1       = 9    # L1       — nav_command LEFT
BTN_R1       = 10   # R1       — nav_command RIGHT

# ==============================
# GRAVEL EDGE DETECTION CONFIG  (HSV color masking)
# ==============================
# The right-lane boundary is a pavement-to-gravel transition.
# We use HSV color masking — same strategy as yellow line detection —
# because gravel and asphalt can have similar brightness (V channel)
# but differ in hue (H) and saturation (S).
#
# HOW TO TUNE THESE VALUES
# -------------------------
#   1. Run V1.13, press N (RIGHT_LANE) and P (debug panel).
#   2. Pause Webots on a straight right-lane section.
#   3. Take a screenshot of the camera frame.
#   4. Open in GIMP (Color Picker tool, set to HSV) or any HSV color picker.
#   5. Sample 5-10 gravel pixels and 5-10 asphalt pixels.
#   6. Set H/S/V min/max to cover the gravel cluster, exclude asphalt.
#   7. Check Gravel Edge Debug window — gravel=white, road=black.
#
# NOTE: OpenCV HSV ranges: H 0-179, S 0-255, V 0-255.
#       Divide a standard 0-360 hue value by 2 for OpenCV.
#
# DEFAULTS: broad initial range — narrow after sampling your world.
EDGE_TARGET_X_RATIO = 0.82   # desired gravel edge x as fraction of frame width
EDGE_CROP_TOP       = 0.70   # ROI top fraction — skip horizon and sky
EDGE_CROP_BOT       = 0.95   # ROI bottom fraction — skip car hood

# HSV color range for gravel/sand — TUNE THESE for your Webots world
GRAVEL_H_MIN =   8   # hue lower bound  (warm sandy/beige tones)
GRAVEL_H_MAX =  35   # hue upper bound
GRAVEL_S_MIN =  15   # saturation lower (slight colour vs neutral gray asphalt)
GRAVEL_S_MAX = 120   # saturation upper (below white-line saturation)
GRAVEL_V_MIN =  60   # brightness lower bound
GRAVEL_V_MAX = 210   # brightness upper bound

# CLAHE equalizes V channel brightness locally so shadowed gravel and
# lit gravel land in the same V range — fixes detection loss under shade.
# clipLimit: higher = stronger equalization (1.0–4.0 typical).
# tileGridSize: size of local regions for equalization (8x8 is standard).
GRAVEL_CLAHE_CLIP  = 2.0      # clip limit for CLAHE
GRAVEL_CLAHE_GRID  = (8, 8)   # tile grid size for CLAHE

# Seconds to keep nav_command active after the lane reference is reacquired
# following an intersection. Extends 4 Hz capture into the post-turn straight.
NAV_CMD_RESET_DELAY = 3.0

#Virtual Lane
LANE_OFFSET_PX = 110

#Blue dot location/ reference to screen
Y_REF_RATIO = 0.85

# CANNY + HOUGH / / Jump rejection to avoid edge-to-edge oscillation
MAX_LINE_JUMP_PX = 100

#Line MAx SLOPE allowed for detection
MAX_SLOPE=0.15

# ==============================
# OBJECT DETECTION CONFIG — Camera Recognition (built-in)
# ==============================
# Replaces SVM-based detection. Uses Webots' built-in neural network.
# Set to True to enable recognition overlay in the debug window.
ENABLE_OBJECT_DETECTION = True

# ==============================
# DEBUG PANEL
# ==============================

# Enables or disables the debug overlay on the camera display.
# Toggle at runtime with key P.
DEBUG_PANEL = True

# ==============================
# DEBUG PRINT (Console)
# ==============================

# Enables or disables verbose console prints (detect models, signs, etc.)
# Useful for troubleshooting camera recognition.
DEBUG_PRINT = False

# ==============================
# ADAS MONITOR
# ==============================

# Enables or disables the ADAS Monitor overlay window.
# Toggle at runtime with key A.
ADAS_MONITOR_ENABLED = True

# ==============================
# LIDAR OBSTACLE DETECTION CONFIG
# ==============================

# Enables or disables LiDAR-based obstacle detection.
ENABLE_LIDAR_OBSTACLE_DETECTION = True

# Maximum detection distance required by the assignment.
# Any obstacle farther than this value will be ignored.
LIDAR_MAX_DISTANCE = 25.0  # meters

# Distance threshold to stop the vehicle.
# This is intentionally smaller than 20 m because 20 m is the detection limit,
# not necessarily the emergency stop distance.
LIDAR_STOP_DISTANCE = 10.0  # meters


# ==============================
# PHASE 2 RETURNING — sensor-based config
# ==============================
# DS_MR (Middle Right) keeps the car away from the right road guard.
# Values depend on the sensor lookup table in the Webots world.
# Tune DS_RETURN_RIGHT_MIN so the car stays ~2 m from the right guard.
DS_RETURN_RIGHT_MIN    = 300   # below this → too close to right guard, steer left
DS_RETURN_RIGHT_TARGET = 600   # ideal right clearance (used for proportional correction)
RETURN_LANE_ANGLE      = -0.3 # base left-return steer (negative = left in this controller)
RETURN_SPEED_RATIO     = 0.40  # fraction of AUTONOMOUS_SPEED used while returning

# Phase 1 EVADING — left sensor clearance to trigger RETURNING
# Sensors read ~1000 when clear (max range), ~400 when obstacle is alongside.
# Latch fires when any sensor drops BELOW this value (obstacle detected).
# Clear fires when ALL sensors rise ABOVE this value (obstacle gone).
# Keep between the obstacle reading (~400) and the max clear reading (~1000).
DS_LEFT_CLEAR_THRESHOLD = 950   # sensor units — tune per world
EVADING_MAX_DURATION    = 10.0  # safety timeout (s): force → RETURNING if sensors never clear

# Phase 1 EVADING — parallel control (DS_ML) once car is alongside the obstacle.
# BC steers right first; once _left_ds_triggered, DS_ML takes over to hold parallel distance.
# DS_ML reads ~400 when very close, ~1000 when clear.
DS_LEFT_PARALLEL_MIN    = 350   # below this → too close to obstacle, nudge right
DS_LEFT_PARALLEL_TARGET = 550   # ideal lateral distance to obstacle (go straight)
PARALLEL_NUDGE_ANGLE    = -0.1  # small right steer when too close (positive = right)

# Phase 2 RETURNING — exit conditions
RETURNING_DURATION   = 5.0   # safety timeout (s): exit RETURNING if wZ never aligns
WZ_RETURN_TOLERANCE  = 0.05  # rad/s: wZ must be within this of saved phase-1 value to exit

# ==============================
# BEHAVIOR CLASS — sharp turn debounce
# ==============================
SHARP_TURN_ENTRY_ANGLE  = 0.25  # rad — must exceed this to enter class 3/4
SHARP_TURN_EXIT_ANGLE   = 0.15  # rad — must drop below this to release back to class 0
SHARP_TURN_DWELL_SECS   = 4.0   # seconds to hold class 3/4 after entry

# ==============================
# BEHAVIORAL CLONING CONFIG
# ==============================

# Path to the 5-class TorchScript model exported from Notebook 27.
# Train with 27-BehaviorClonning_BehaviorClass.ipynb before running V1.12.
_BC_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "bc_behavior_model.pt"
)

# Initial state for the two independent runtime toggles (changed with B / L keys).
# Both ON = HYBRID mode (recommended starting point).
BC_ENABLED_DEFAULT  = True   # B key: BC inference on/off
PID_ENABLED_DEFAULT = True   # L key: PID / line-follower on/off

# BC preprocessing constants — must match Notebook 27 exactly.
BC_IMG_H    = 66
BC_IMG_W    = 200
BC_CROP_TOP = 0.38   # fraction of frame height to crop from top (sky)
BC_CROP_BOT = 0.88   # fraction of frame height where crop ends (removes hood)

# EMA smoothing factor for BC output.
# 0.0 = frozen (never updates), 1.0 = raw output (no smoothing).
# 0.40 at a 32 ms timestep gives ~80 ms effective lag — removes single-frame
# jitter without slowing the steering response to real road curves.
BC_SMOOTH_ALPHA = 0.40

# Maximum absolute angle the BC model is allowed to output (radians).
# Matches the training data range: dataset min/max is approx ±0.70 rad.
# Guards against extreme predictions on frames outside the training distribution.
BC_ANGLE_CLAMP = 0.70

# Per-class target cruising speed in BC mode (km/h).
# Classes 3 and 4 are slower because sharp turns require precise low-speed control.
# Edit these values to tune per-track without touching the logic below.
BC_SPEED_BY_CLASS = {
    0: 30.0,   # NORMAL        — steady road following
    1: 30.0,   # EVADING       — keep momentum while passing a vehicle
    2: 30.0,   # RETURNING     — merge back at a comfortable speed
    3: 15.0,   # SHARP_LEFT    — slow entry into left-turn intersection
    4: 15.0,   # SHARP_RIGHT   — slow entry into right-turn intersection
}

# Stuck-vehicle detection thresholds.
# If driver.getCurrentSpeed() stays below BC_STUCK_SPEED_THRESHOLD for longer
# than BC_STUCK_TIME_THRESHOLD seconds while BC commands movement, a warning is
# printed. The timer resets after each warning so the alert repeats every 5 s.
BC_STUCK_SPEED_THRESHOLD = 5.0   # km/h — below this = "not moving"
BC_STUCK_TIME_THRESHOLD  = 5.0   # seconds before warning fires


_BC_DEVICE  = None   # set once at load time, used by all BC inference calls
_CIL_DEVICE = None   # set once at load time, used by all CIL inference calls

# ==============================
# CIL INFERENCE CONFIG
# ==============================

# Path to the TorchScript CIL model exported from Notebook 28.
_CIL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "cil_model.pt"
)

# Speed (km/h) while CIL is steering through an intersection.
# Lower than AUTONOMOUS_SPEED so turns are precise.
CIL_SPEED = 15.0

# EMA smoothing factor — same convention as BC_SMOOTH_ALPHA.
CIL_SMOOTH_ALPHA = 0.40

# Hard clamp on raw CIL output to guard against OOD frames.
CIL_ANGLE_CLAMP = 0.70

# Preprocessing — MUST match CILDataset._load_and_crop in Notebook 28 exactly.
CIL_CROP_TOP  = 0.40          # fraction from top to crop (sky)
CIL_CROP_BOT  = 0.95          # fraction where crop ends (removes hood)
CIL_IMG_W     = 320
CIL_IMG_H     = 160
CIL_NORM_MEAN = [0.485, 0.456, 0.406]
CIL_NORM_STD  = [0.229, 0.224, 0.225]

# Human-readable names for nav_command values (used in HUD and console)
_CIL_CMD_NAMES = {
    dataset_mode.CMD_LANE_FOLLOW: "LANE_FOLLOW",
    dataset_mode.CMD_STRAIGHT:    "STRAIGHT",
    dataset_mode.CMD_LEFT:        "LEFT",
    dataset_mode.CMD_RIGHT:       "RIGHT",
}


def _bc_load_model():
    """
    Load the TorchScript BC model at controller startup.
    Returns the model object, or None if unavailable (torch missing, file not found).
    Failure is non-fatal: the controller continues with PID only.
    """
    global _BC_DEVICE
    if not _BC_TORCH_AVAILABLE:
        return None
    if not os.path.isfile(_BC_MODEL_PATH):
        print(f"[BC] Model file not found: {_BC_MODEL_PATH}")
        print("[BC] Run Notebook 27 (27-BehaviorClonning_BehaviorClass.ipynb) to train and export the model.")
        print("[BC] Controller will run in PID-only mode until the model is available.")
        return None
    _BC_DEVICE = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = _torch.jit.load(_BC_MODEL_PATH, map_location=_BC_DEVICE)
    model.eval()
    print(f"[BC] Model loaded: {_BC_MODEL_PATH}")
    print(f"[BC] Inference device: {_BC_DEVICE}")
    return model


def _bc_predict(model, frame_bgr):
    """
    Run one raw BC inference step.
    Crops and resizes the frame to match the training preprocessing in Notebook 27,
    then returns the raw predicted steering angle in radians (unclamped, unsmoothed).
    """
    h, w = frame_bgr.shape[:2]
    y1 = int(h * BC_CROP_TOP)
    y2 = int(h * BC_CROP_BOT)
    crop    = frame_bgr[y1:y2, :]
    resized = cv2.resize(crop, (BC_IMG_W, BC_IMG_H))
    img     = resized.astype(np.float32) / 255.0
    t = _torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(_BC_DEVICE)
    with _torch.no_grad():
        return float(model(t).item())


def _bc_predict_smooth(model, frame_bgr, prev_smoothed):
    """
    Run BC inference and return a clamped, EMA-smoothed steering angle.

    Steps:
      1. Raw inference via _bc_predict.
      2. Clamp to ± BC_ANGLE_CLAMP so spikes don't pollute the EMA history.
      3. Apply Exponential Moving Average with factor BC_SMOOTH_ALPHA.

    Parameters
    ----------
    model          Loaded TorchScript model (from _bc_load_model).
    frame_bgr      Current camera frame in BGR format (full resolution).
    prev_smoothed  EMA state from the previous frame (initialise to 0.0).

    Returns
    -------
    float — smoothed angle ready to be applied to driver.setSteeringAngle().
    """
    raw     = _bc_predict(model, frame_bgr)
    raw     = max(-BC_ANGLE_CLAMP, min(BC_ANGLE_CLAMP, raw))          # clamp first
    return BC_SMOOTH_ALPHA * raw + (1.0 - BC_SMOOTH_ALPHA) * prev_smoothed  # EMA


# ==============================
# CIL INFERENCE FUNCTIONS
# ==============================

def _cil_load_model():
    """
    Load the TorchScript CIL model at controller startup.
    Returns the model object, or None if unavailable (torch missing, file not found).
    Failure is non-fatal: nav_cmd ≠ LANE_FOLLOW falls back to BC/PID arbitration.
    """
    global _CIL_DEVICE
    if not _BC_TORCH_AVAILABLE:
        return None
    if not os.path.isfile(_CIL_MODEL_PATH):
        print(f"[CIL] Model file not found: {_CIL_MODEL_PATH}")
        print("[CIL] Run Notebook 28 (28-BehavioralCloning_CIL.ipynb) to train and export.")
        print("[CIL] nav_cmd ≠ LANE_FOLLOW will fall back to BC/PID until model is available.")
        return None
    _CIL_DEVICE = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = _torch.jit.load(_CIL_MODEL_PATH, map_location=_CIL_DEVICE)
    model.eval()
    print(f"[CIL] Model loaded : {_CIL_MODEL_PATH}")
    print(f"[CIL] Device       : {_CIL_DEVICE}")
    return model


def _cil_preprocess(frame_bgr):
    """
    Preprocess a camera frame for CIL inference.
    Must match CILDataset._load_and_crop in Notebook 28 exactly:
      1. Crop: remove top CIL_CROP_TOP and keep until CIL_CROP_BOT.
      2. Convert BGR → RGB.
      3. Resize to (CIL_IMG_W, CIL_IMG_H).
      4. Normalize to float32 [0,1] then apply ImageNet mean/std.
    Returns a (1, 3, CIL_IMG_H, CIL_IMG_W) tensor on _CIL_DEVICE.
    """
    h, w = frame_bgr.shape[:2]
    y1   = int(h * CIL_CROP_TOP)
    y2   = int(h * CIL_CROP_BOT)
    crop = frame_bgr[y1:y2, :]
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (CIL_IMG_W, CIL_IMG_H))
    t    = _torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = _torch.tensor(CIL_NORM_MEAN).view(3, 1, 1)
    std  = _torch.tensor(CIL_NORM_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(_CIL_DEVICE)   # (1, 3, H, W)


def _cil_predict_smooth(model, frame_bgr, nav_command, prev_smoothed):
    """
    Run one CIL inference step and return a clamped, EMA-smoothed steering angle.

    Parameters
    ----------
    model         Loaded TorchScript CIL model (from _cil_load_model).
    frame_bgr     Current camera frame in BGR format (full resolution).
    nav_command   Integer command (1=STRAIGHT, 2=LEFT, 3=RIGHT).
    prev_smoothed EMA state from the previous frame (initialise to 0.0).

    Returns
    -------
    float — smoothed angle ready to be applied to driver.setSteeringAngle().
    """
    tensor = _cil_preprocess(frame_bgr)
    with _torch.no_grad():
        raw = float(model(tensor, nav_command).item())
    raw = max(-CIL_ANGLE_CLAMP, min(CIL_ANGLE_CLAMP, raw))
    return CIL_SMOOTH_ALPHA * raw + (1.0 - CIL_SMOOTH_ALPHA) * prev_smoothed


# ==============================
# PID CONTROLLER
# ==============================

class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

        self.previous_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        # Guard against zero/negative dt (first frame or timer glitch)
        if dt <= 0:
            dt = 0.001

        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.previous_error = error

        # Clamp to symmetric output limit (maps to MAX_ANGLE in steering)
        output = max(-self.output_limit, min(self.output_limit, output))

        return output


# ==============================
# HELPER FUNCTIONS
# ==============================

def apply_deadzone(value, deadzone=DEADZONE):
    if abs(value) < deadzone:
        return 0.0
    return value


def get_image(camera):
    raw_image = camera.getImage()

    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )

    # Webots camera image is BGRA
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return frame_bgr


def display_image(display, image_bgr):
    # Webots Display expects RGB bytes; convert from OpenCV's native BGR first
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )

    display.imagePaste(image_ref, 0, 0, False)


def init_ps_controller():
    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()

    if joystick_count == 0:
        print("No PS controller detected. Keyboard only mode.")
        return None

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Controller detected: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}, Buttons: {joystick.get_numbuttons()}")

    return joystick


def smooth_line(current_line, previous_line, alpha=0.25):
    # Exponential moving average across the four line endpoints [x1,y1,x2,y2].
    # Low alpha (0.25) weights the history heavily, dampening frame-to-frame jitter.
    if current_line is None:
        return previous_line

    if previous_line is None:
        return current_line

    return [
        int(alpha * current_line[0] + (1 - alpha) * previous_line[0]),
        int(alpha * current_line[1] + (1 - alpha) * previous_line[1]),
        int(alpha * current_line[2] + (1 - alpha) * previous_line[2]),
        int(alpha * current_line[3] + (1 - alpha) * previous_line[3])
    ]


class PIDDebugChart:
    # [ADDED] error_range: half the camera pixel width used to scale the error
    # signal. Default 320 matches a 620-wide camera. Pass camera.getWidth()//2
    # at construction so the scale stays correct if the resolution ever changes.
    # [CHANGED] Default size set to 640x300 to match the "Self Driving Debug"
    # and "Yellow Mask Debug" windows so all three charts sit at the same scale.
    def __init__(self, width=640, height=300, max_points=200, error_range=320):
        self.width = width
        self.height = height
        self.max_points = max_points
        self.error_range = error_range

        self.errors = []
        self.angles = []
        self.speeds = []

        # [ADDED] Pre-allocate the canvas once and reuse it every frame by
        # calling fill(0) in show() instead of np.zeros(). This avoids a heap
        # allocation on every simulation step.
        self._canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def update(self, error, angle, speed):
        self.errors.append(error)
        self.angles.append(angle)
        self.speeds.append(speed)

        self.errors = self.errors[-self.max_points:]
        self.angles = self.angles[-self.max_points:]
        self.speeds = self.speeds[-self.max_points:]

    def normalize(self, values, min_val, max_val):
        if max_val == min_val:
            return [self.height // 2 for _ in values]

        # [FIXED] Clamp each pixel coordinate to [0, height-1].
        # Without clamping, any value outside [min_val, max_val] (e.g. a speed
        # spike above 50 or an error beyond the image edge) produces a negative
        # or out-of-bounds y coordinate. OpenCV silently skips those line
        # segments, making part of the signal vanish from the chart with no
        # visible error. Clamping keeps everything inside the canvas.
        return [
            max(0, min(self.height - 1,
                int(self.height - ((v - min_val) / (max_val - min_val)) * self.height)
            ))
            for v in values
        ]

    # [CHANGED] Added label_index (int) as an explicit parameter to control
    # the vertical position of the legend text. The original code used
    # label[0], which returns a str character ("1", "2", "3") in Python 3.
    # 25 * "1" produces a repeated string, and 25 + that string raises a
    # TypeError crash on the second simulation step (first frame with >= 2
    # data points). Passing label_index as a plain int fixes this completely.
    def draw_signal(self, img, values, min_val, max_val, color, label, label_index):
        if len(values) < 2:
            return

        y_values = self.normalize(values, min_val, max_val)

        # [ADDED] Draw a per-signal zero reference line.
        # The original code drew a single shared center line at height//2, which
        # only made visual sense for signals symmetric around zero (Error and
        # Steering). Speed lives in [0, 50], so its true zero is at the bottom
        # of the chart — the old center line crossed it at 25 km/h, making 25
        # look like zero. Each signal now gets its own zero line placed at the
        # correct pixel position using normalize(), so all three traces are
        # independently readable.
        zero_y = self.normalize([0], min_val, max_val)[0]
        cv2.line(img, (0, zero_y), (self.width, zero_y), (60, 60, 60), 1)

        for i in range(1, len(values)):
            x1 = int((i - 1) * self.width / self.max_points)
            x2 = int(i * self.width / self.max_points)

            cv2.line(
                img,
                (x1, y_values[i - 1]),
                (x2, y_values[i]),
                color,
                2
            )

        cv2.putText(
            img,
            label,
            (10, 25 + 25 * label_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    def show(self):
        # [CHANGED] Reuse the pre-allocated self._canvas instead of creating a
        # new np.zeros array every frame. fill(0) zeros the buffer in-place,
        # which is significantly cheaper than a fresh allocation each step.
        self._canvas.fill(0)

        # [REMOVED] The single shared center line has been removed here.
        # Each signal now draws its own zero reference line inside draw_signal(),
        # correctly placed for its own value range (see notes there).

        # Error signal: pixel offset between the detected yellow line and the
        # image center. Range uses self.error_range (= camera.getWidth()//2)
        # so the scale stays accurate if the camera resolution is changed.
        self.draw_signal(
            self._canvas,
            self.errors,
            min_val=-self.error_range,
            max_val=self.error_range,
            color=(0, 255, 255),
            label="Error (px)",
            label_index=1
        )

        # Steering angle output from the PID controller, in radians.
        # Range matches STEERING_GAIN_LIMIT = 0.5.
        self.draw_signal(
            self._canvas,
            self.angles,
            min_val=-0.5,
            max_val=0.5,
            color=(0, 255, 0),
            label="Steering (rad)",
            label_index=2
        )

        # Cruising speed in km/h. Upper bound set to cover AUTONOMOUS_SPEED
        # and manual CONTROLLER_MAX_SPEED with some headroom.
        self.draw_signal(
            self._canvas,
            self.speeds,
            min_val=0,
            max_val=100,
            color=(255, 0, 255),
            label="Speed (km/h)",
            label_index=3
        )

        cv2.imshow("PID Response Chart", self._canvas)


# ==============================
# ADAS MONITOR
# ==============================

def draw_adas_monitor(speed, angle, brake, gps_vals, gyro_vals,
                      ds_ml_val, ds_mr_val, ds_fl_val, ds_fr_val, ds_rl_val, ds_rr_val,
                      autonomous_mode=False, evasion_phase=0, line_detected=True,
                      bc_active=False, lidar_stop=False, lidar_ahead=False,
                      lidar_obstacle=False, lidar_distance=None, lidar_angle=None,
                      obstacle_type="", vehicle_nearby=False, parallel_active=False,
                      saved_gyro_wz=None, current_gyro_wz=None,
                      bc_enabled=True, pid_enabled=True,
                      lane_side=0, edge_detected=False, nav_command=0,
                      cil_active=False, cil_model_loaded=False):
    # Panel height: V1.11=833, V1.12=879 (+46), V1.13=948 (+69), V1.14=971 (+23 CIL row)
    W, H = 400, 971
    panel = np.zeros((H, W, 3), dtype=np.uint8)

    def section_header(y, title):
        cv2.putText(panel, title, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)
        cv2.line(panel, (8, y + 6), (W - 8, y + 6), (50, 50, 50), 1)

    def data_row(y, label, value, val_color=(210, 210, 210)):
        cv2.putText(panel, label, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 130, 130), 1)
        cv2.putText(panel, value, (175, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, val_color, 1)

    def reserved_row(y, label):
        cv2.putText(panel, label, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (55, 55, 55), 1)

    def ds_color(val):
        if val is None:
            return (55, 55, 55)
        if val < 500:
            return (0, 0, 255)
        if val < 1000:
            return (0, 165, 255)
        return (0, 255, 0)

    # Title
    cv2.putText(panel, "ADAS  MONITOR", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(panel, (8, 32), (W - 8, 32), (80, 80, 80), 1)

    # ── MACHINE STATE ─────────────────────────────────
    section_header(50, "MACHINE STATE")

    # Mode
    if autonomous_mode:
        mode_str, mode_color = "AUTONOMOUS", (0, 255, 255)
    else:
        mode_str, mode_color = "MANUAL", (0, 255, 0)
    data_row(72, "Mode", mode_str, mode_color)

    # Phase
    _phase_map = {
        0: ("0 - NORMAL",          (0, 255, 0)),
        1: ("1 - EVADING",         (0, 165, 255)),
        2: ("2 - RETURNING",       (0, 200, 255)),
        3: ("3 - SHARP_TURN_LEFT", (255, 200, 0)),
        4: ("4 - SHARP_TURN_RIGHT",(255, 200, 0)),
    }
    phase_str, phase_color = _phase_map.get(evasion_phase, (str(evasion_phase), (210, 210, 210)))
    if evasion_phase == 0:
        if lidar_stop:
            phase_str, phase_color = "EMERGENCY STOP", (0, 0, 255)
        elif lidar_ahead:
            phase_str, phase_color = "OBJECT AHEAD",   (0, 100, 255)
    data_row(95, "Phase", phase_str, phase_color)

    # Active controller
    if lidar_stop:
        ctrl_str, ctrl_color = "BRAKE (full)", (0, 0, 255)
    elif evasion_phase == 1 and parallel_active:
        ctrl_str, ctrl_color = "SENSOR  DS_ML", (0, 200, 100)
    elif evasion_phase == 1 and bc_active:
        ctrl_str, ctrl_color = "BC  EVADING", (0, 165, 255)
    elif evasion_phase == 2:
        ctrl_str, ctrl_color = "SENSOR  DS_MR", (0, 200, 255)
    elif cil_active:
        _cil_ctrl_name = {1: "STRAIGHT", 2: "LEFT", 3: "RIGHT"}.get(nav_command, "?")
        ctrl_str, ctrl_color = f"CIL  {_cil_ctrl_name}", (255, 200, 0)
    elif not line_detected and bc_active:
        ctrl_str, ctrl_color = "BC  LINE_LOST", (255, 165, 0)
    elif not line_detected:
        ctrl_str, ctrl_color = "DECAY  fallback", (0, 80, 255)
    elif autonomous_mode:
        ctrl_str, ctrl_color = "PID  lane follow", (0, 255, 255)
    else:
        ctrl_str, ctrl_color = "MANUAL  input", (0, 255, 0)
    data_row(118, "Control", ctrl_str, ctrl_color)

    # ── BC / PID status rows (new in V1.12) ───────────────────────────────
    # BC row — shows which mode is active based on both toggle states
    if bc_enabled and pid_enabled:
        bc_row_str, bc_row_color = "ON  (hybrid: BC=1/2/3/4  PID=0)", (0, 200, 255)
    elif bc_enabled and not pid_enabled:
        bc_row_str, bc_row_color = "ON  (full — all classes)",         (0, 165, 255)
    elif not bc_enabled and pid_enabled:
        bc_row_str, bc_row_color = "OFF (pure PID mode)",              (0, 255, 0)
    else:
        bc_row_str, bc_row_color = "OFF (safe hold — both disabled)",  (0, 0, 255)
    data_row(141, "BC model", bc_row_str, bc_row_color)

    # PID row — shows line-follower state
    if pid_enabled:
        pid_row_str, pid_row_color = "ON",                    (0, 255, 0)
    else:
        pid_row_str, pid_row_color = "OFF (BC handles all)", (0, 165, 255)
    data_row(164, "PID / line", pid_row_str, pid_row_color)

    # CIL row — new in V1.14
    if cil_active:
        cil_row_str   = f"ACTIVE  ({_CIL_CMD_NAMES.get(nav_command, '?')})"
        cil_row_color = (255, 200, 0)
    elif cil_model_loaded:
        cil_row_str, cil_row_color = "LOADED  (standby)", (0, 200, 100)
    else:
        cil_row_str, cil_row_color = "NOT FOUND  (fallback→BC/PID)", (0, 80, 255)
    data_row(187, "CIL model", cil_row_str, cil_row_color)

    # ── V1.13 rows — shifted +23 px from V1.13 positions ─────────────────

    # Lane side — which lane and reference is active
    if lane_side == 1:  # LANE_RIGHT
        ls_str, ls_color = "RIGHT  (gravel edge)", (0, 200, 100)
    else:
        ls_str, ls_color = "LEFT   (yellow line)", (0, 255, 255)
    data_row(210, "Lane side", ls_str, ls_color)

    # Gravel edge detection status
    if edge_detected:
        ed_str, ed_color = "DETECTED", (0, 255, 0)
    else:
        ed_str, ed_color = "NOT FOUND", (0, 80, 255)
    data_row(233, "Gravel edge", ed_str, ed_color)

    # Navigation command — CIL intent set by PS4 L1/R1/O
    _nav_labels = {0: "LANE_FOLLOW", 1: "STRAIGHT", 2: "LEFT", 3: "RIGHT"}
    _nav_colors = {
        0: (100, 100, 100),
        1: (0, 255, 0),
        2: (255, 200, 0) if cil_active else (0, 165, 255),
        3: (255, 200, 0) if cil_active else (0, 165, 255),
    }
    data_row(256, "Nav cmd", _nav_labels.get(nav_command, "?"),
             _nav_colors.get(nav_command, (210, 210, 210)))

    # Yellow line
    if evasion_phase in (1, 2):
        line_str, line_color = "N/A (off-lane)", (100, 100, 100)
    elif line_detected:
        line_str, line_color = "DETECTED", (0, 255, 0)
    else:
        line_str, line_color = "LOST", (0, 0, 255)
    data_row(279, "Yellow line", line_str, line_color)

    # wZ alignment row — only visible during RETURNING
    if evasion_phase == 2 and saved_gyro_wz is not None and current_gyro_wz is not None:
        diff = abs(current_gyro_wz - saved_gyro_wz)
        wz_color = (0, 255, 0) if diff <= WZ_RETURN_TOLERANCE else (0, 165, 255)
        wz_str = f"ref:{saved_gyro_wz:.3f}  cur:{current_gyro_wz:.3f}"
        data_row(302, "wZ align", wz_str, wz_color)

    cv2.line(panel, (8, 316), (W - 8, 316), (50, 50, 50), 1)

    spd_color = (0, 255, 0) if speed < 60 else (0, 165, 255)
    brk_color = (0, 0, 255) if brake > 0.1 else (210, 210, 210)
    data_row(351, "Speed",  f"{speed:.1f}  km/h", spd_color)
    data_row(374, "Angle",  f"{angle:.4f}  rad")
    data_row(397, "Brake",  f"{brake:.2f}", brk_color)

    # ── GPS ───────────────────────────────────────────────────────────────
    section_header(425, "GPS")
    if gps_vals is not None:
        data_row(447, "X", f"{gps_vals[0]:.3f}  m")
        data_row(470, "Y", f"{gps_vals[1]:.3f}  m")
        data_row(493, "Z", f"{gps_vals[2]:.3f}  m")
    else:
        data_row(457, "Status", "NOT AVAILABLE", (0, 80, 255))

    # ── GYRO ──────────────────────────────────────────────────────────────
    section_header(521, "GYRO")
    if gyro_vals is not None:
        data_row(543, "wX", f"{gyro_vals[0]:.4f}  rad/s")
        data_row(566, "wY", f"{gyro_vals[1]:.4f}  rad/s")
        data_row(589, "wZ", f"{gyro_vals[2]:.4f}  rad/s")
    else:
        data_row(553, "Status", "NOT AVAILABLE", (0, 80, 255))

    # ── DISTANCE SENSORS ──────────────────────────────────────────────────
    section_header(617, "DISTANCE SENSORS")
    ml_str = f"{ds_ml_val:.1f}" if ds_ml_val is not None else "--"
    mr_str = f"{ds_mr_val:.1f}" if ds_mr_val is not None else "--"
    fl_str = f"{ds_fl_val:.1f}" if ds_fl_val is not None else "--"
    fr_str = f"{ds_fr_val:.1f}" if ds_fr_val is not None else "--"
    rl_str = f"{ds_rl_val:.1f}" if ds_rl_val is not None else "--"
    rr_str = f"{ds_rr_val:.1f}" if ds_rr_val is not None else "--"
    data_row(639, "DS_ML  (mid-left)",    ml_str, ds_color(ds_ml_val))
    data_row(662, "DS_MR  (mid-right)",   mr_str, ds_color(ds_mr_val))
    data_row(685, "DS_FL  (front-left)",  fl_str, ds_color(ds_fl_val))
    data_row(708, "DS_FR  (front-right)", fr_str, ds_color(ds_fr_val))
    data_row(731, "DS_RL  (rear-left)",   rl_str, ds_color(ds_rl_val))
    data_row(754, "DS_RR  (rear-right)",  rr_str, ds_color(ds_rr_val))

    # ── LIDAR ─────────────────────────────────────────────────────────────
    section_header(782, "LIDAR  (Sick LMS 291)")

    if lidar_obstacle and lidar_distance is not None:
        dist_color = (0, 0, 255) if lidar_distance < LIDAR_STOP_DISTANCE else (0, 165, 255)
        data_row(804, "Status",   "OBSTACLE DETECTED", dist_color)
        data_row(827, "Distance", f"{lidar_distance:.2f}  m", dist_color)
        ang_str = f"{lidar_angle:.1f}  deg" if lidar_angle is not None else "--"
        data_row(850, "Angle",    ang_str, (210, 210, 210))
        type_color = (0, 100, 255) if obstacle_type == "VEHICLE" else (210, 210, 210)
        data_row(873, "Type",     obstacle_type if obstacle_type else "--", type_color)
        vh_str = "YES" if vehicle_nearby else "NO"
        vh_color = (0, 0, 255) if vehicle_nearby else (0, 200, 0)
        data_row(896, "Vehicle",  vh_str, vh_color)
    else:
        data_row(804, "Status",   "CLEAR", (0, 255, 0))
        data_row(827, "Distance", "--")
        data_row(850, "Angle",    "--")
        data_row(873, "Type",     "--")
        data_row(896, "Vehicle",  "NO", (0, 200, 0))

    # Bottom border
    cv2.line(panel, (8, H - 8), (W - 8, H - 8), (50, 50, 50), 1)

    cv2.imshow("ADAS Monitor", panel)


# ==============================
# YELLOW LINE DETECTION
# ==============================

def detect_yellow_line(frame):
    global yellow_line_prev, line_x_history

    height, width, _ = frame.shape
    debug_frame = frame.copy()

    # ==============================
    # 1. GRAYSCALE BRIGHTNESS MASK
    # Replaces the HSV yellow mask used in V1.2SDF_.
    # Grayscale thresholding is faster and avoids hue-shift sensitivity, but
    # it will also pick up any other bright object in the ROI (white markings,
    # reflections). The narrow ROI in step 2 limits false positives.
    # ==============================

    # Convert BGR to single-channel grayscale.
    # Unlike the HSV path, all three color channels are collapsed into one
    # luminance value, so cv2.inRange uses scalar (not array) thresholds.
    GRAY_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Brightness thresholds: keep only bright pixels (120–255).
    # The yellow road centerline in Webots is consistently bright (~180–255).
    # A lower bound of 120 excludes the dark road surface while still catching
    # the line under varying virtual lighting conditions.
    lower_yellow = 120
    upper_yellow = 200

    # cv2.inRange on a single-channel image with scalar bounds produces a
    # binary mask (255 where in range, 0 elsewhere) — same output type as the
    # HSV inRange used in V1.2SDF_, so all downstream steps are unchanged.
    yellow_mask = cv2.inRange(GRAY_, lower_yellow, upper_yellow)

    # ==============================
    # 2. NARROW ROI
    # Focus only where the centerline should appear
    # ==============================
    
    roi_mask = np.zeros_like(yellow_mask)

    vertices = np.array([[
        (int(width * 0.1), int(height * 1.00)),
        (int(width * 0.1), int(height * 0.55)),
        (int(width * 0.65), int(height * 0.75)),
        (int(width * 0.65), int(height * 1.00))
    ]], dtype=np.int32)

    cv2.fillPoly(roi_mask, vertices, 255)
    yellow_roi = cv2.bitwise_and(yellow_mask, roi_mask)

    # Clean mask with two-pass morphology.
    # OPEN (erode → dilate) removes small isolated bright specks that the
    # grayscale threshold picks up more aggressively than the HSV mask did
    # (e.g. windshield glare, road texture highlights).
    # CLOSE (dilate → erode) fills small gaps inside the line blob so
    # Canny/Hough see a continuous edge rather than a fragmented one.
    # DILATE thickens the remaining blob to give Hough more edge pixels to vote on.
    kernel = np.ones((5, 5), np.uint8)
    yellow_roi = cv2.morphologyEx(yellow_roi, cv2.MORPH_OPEN, kernel)
    yellow_roi = cv2.morphologyEx(yellow_roi, cv2.MORPH_CLOSE, kernel)
    yellow_roi = cv2.dilate(yellow_roi, kernel, iterations=1)

    # ==============================
    # 3. CANNY + HOUGH
    # ==============================
    blur = cv2.GaussianBlur(yellow_roi, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=2,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=30,
        maxLineGap=80
    )

    desired_x = int(width * 0.50)
    # desired_x magenta reference line is drawn in the main loop (elif line_detected block)
    # so it only appears in LEFT_LANE mode and not when gravel edge is active.

    # Testing value to adjust the position of the reference to the car.
    # Adjusting to close it to the car means a higher number, to improve turning. 
    y_ref = int(height * Y_REF_RATIO)

    valid_lines = []
    slopes = []
    intercepts = []
    weights = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:

                if x2 == x1:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                length = np.hypot(x2 - x1, y2 - y1)

                if length < 30:
                    continue

                # Ignore almost horizontal lines <0.25
                # Yellow lane should be mostly vertical in the camera view <1.2
                if abs(slope) < MAX_SLOPE:
                    continue

                intercept = y1 - slope * x1

                # Find x where this Hough line crosses y_ref
                x_at_ref = int((y_ref - intercept) / slope)

                # Keep only reasonable center-area intersections
                if width * 0.1 < x_at_ref < width * 0.75:
                    slopes.append(slope)
                    intercepts.append(intercept)
                    weights.append(length)
                    valid_lines.append((x1, y1, x2, y2))

    line_detected = False
    line_x_reference = None

    if len(slopes) > 0:
        avg_slope = np.average(slopes, weights=weights)
        avg_intercept = np.average(intercepts, weights=weights)

        line_x_reference = int((y_ref - avg_intercept) / avg_slope)

        # Reject impossible detections far from expected yellow-line area
        if not (width * 0.05 < line_x_reference < width * 0.55):
            line_detected = False
            yellow_line_prev = None
            line_x_history.clear()
            return line_detected, None, desired_x, debug_frame, yellow_roi

        
        # Reject sudden jumps
        if yellow_line_prev is not None:
            prev_x = yellow_line_prev[0]

            if abs(line_x_reference - prev_x) > MAX_LINE_JUMP_PX:
                line_detected = False
                line_x_history.clear()
                return line_detected, None, desired_x, debug_frame, yellow_roi

        
        # Moving average across last 5 valid frame detections
        line_x_history.append(line_x_reference)
        line_x_reference = int(np.mean(line_x_history))

        # Build stable vertical reference line
        yellow_current = [
        line_x_reference,
        int(height * 0.98),
        line_x_reference,
        int(height * 0.55)
        ]

        # Temporal smoothing
        yellow_line_prev = smooth_line(
        yellow_current,
        yellow_line_prev,
        alpha_smooth
        )

        x1, y1, x2, y2 = yellow_line_prev
        line_x_reference = x1

        # Draw all raw Hough lines lightly
        for lx1, ly1, lx2, ly2 in valid_lines:
            cv2.line(debug_frame, (lx1, ly1), (lx2, ly2), (0, 180, 180), 2)


        # Draw final center reference from Hough result
        cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 8)

        # Reference point used by PID
        cv2.circle(debug_frame, (line_x_reference, y_ref), 9, (255, 0, 0), -1)

        line_detected = True

    else:
        yellow_line_prev = None
        line_x_history.clear()

    return line_detected, line_x_reference, desired_x, debug_frame, yellow_roi

# ==============================
# SIMPLE LIDAR PROCESSING
# ==============================
# GRAVEL EDGE DETECTION  (right-lane reference)
# ==============================

def detect_gravel_edge(frame):
    """
    Detect the pavement-to-gravel boundary using HSV color masking.

    Gravel and asphalt often share similar brightness but differ in hue and
    saturation — gravel has a warm sandy/beige hue; asphalt is neutral gray.
    HSV masking isolates the gravel region; we find its leftmost column as
    the road edge reference for right-lane PID control.

    Tune GRAVEL_H/S/V_MIN/MAX in the config section above until the
    Gravel Edge Debug window shows gravel=white and asphalt=black.

    Returns
    -------
    detected : bool    — True if a gravel region was found in the ROI.
    edge_x   : int     — x pixel of the leftmost gravel column (road edge).
    error    : float   — signed error from EDGE_TARGET_X_RATIO target.
                         Positive → edge right of target → car drifted left → steer right.
                         Negative → edge left of target  → car drifted right → steer left.
    gravel_mask: ndarray — binary mask for the Gravel Edge Debug window.
    """
    h, w = frame.shape[:2]
    y1 = int(h * EDGE_CROP_TOP)
    y2 = int(h * EDGE_CROP_BOT)
    roi = frame[y1:y2, :]

    # Convert to HSV, then apply CLAHE to the V channel before masking.
    # CLAHE locally normalizes brightness — shadowed gravel becomes relatively
    # brighter so it stays within GRAVEL_V_MIN/MAX alongside lit gravel.
    # H and S are unaffected, so the color discrimination is preserved.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=GRAVEL_CLAHE_CLIP, tileGridSize=GRAVEL_CLAHE_GRID)
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])

    lower = np.array([GRAVEL_H_MIN, GRAVEL_S_MIN, GRAVEL_V_MIN], dtype=np.uint8)
    upper = np.array([GRAVEL_H_MAX, GRAVEL_S_MAX, GRAVEL_V_MAX], dtype=np.uint8)
    gravel_mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup — remove isolated noise pixels and
    # close small gaps so the gravel region is one solid blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gravel_mask = cv2.morphologyEx(gravel_mask, cv2.MORPH_OPEN,  kernel)
    gravel_mask = cv2.morphologyEx(gravel_mask, cv2.MORPH_CLOSE, kernel)

    # Find the leftmost column that contains any gravel pixels —
    # this is the pavement-to-gravel boundary (the road edge)
    cols_with_gravel = np.where(gravel_mask.any(axis=0))[0]

    if len(cols_with_gravel) == 0:
        return False, 0, 0.0, gravel_mask

    edge_x   = int(cols_with_gravel[0])
    target_x = int(w * EDGE_TARGET_X_RATIO)
    error    = float(edge_x - target_x)

    return True, edge_x, error, gravel_mask


# Based directly on course notes
# ==============================

def process_lidar_data(lidar):

    # Read full LiDAR range image.
    # Example:
    # 180 resolution = 180 distance samples.
    range_image = lidar.getRangeImage()

    # LiDAR horizontal resolution.
    lidar_width = lidar.getHorizontalResolution()

    # Center index of the scan.
    center = lidar_width // 2

    # ==========================================
    # INSPECTION WINDOW
    # ==========================================
    # 20 degrees per side.
    # Total inspected area = 40 degrees.
    # ==========================================

    HALF_AREA = 20

    obstacle_detected = False

    min_distance = 999

    obstacle_index = center

    # Scan only front-center LiDAR sector.
    for i in range(center - HALF_AREA, center + HALF_AREA):

        distance = range_image[i]

        # Ignore invalid values.
        if np.isfinite(distance):

            # Ignore anything farther than 30 meters.
            if distance < LIDAR_MAX_DISTANCE:

                obstacle_detected = True

                # Keep closest obstacle.
                if distance < min_distance:
                    min_distance = distance
                    obstacle_index = i

    # No obstacle found.
    if not obstacle_detected:
        return False, None, None

    # Returns the obstacle position as an index offset from center (integer).
    # For the Sick LMS 291 at 180-sample resolution over 180°, 1 index ≈ 1 degree.
    # Do NOT call np.degrees() on this value — it is already in degree-equivalent units.
    obstacle_angle = obstacle_index - center

    return True, obstacle_angle, min_distance

# ==============================
# MAIN CONTROLLER
# ==============================

def main():
    global DEBUG_PANEL, ADAS_MONITOR_ENABLED
    speed = 0
    angle = 0.0
    brake = 0.0
    previous_angle = 0.0   # seed value for the steering smoothing
    last_press = {}

    autonomous_mode = False

    # Tracks the moment the yellow line first went missing in autonomous mode.
    # None means the line is currently visible (or we are in manual mode).
    # When the line is lost, this is set to current_time and a 5-second
    # countdown begins; at zero the controller reverts to manual mode.
    line_lost_time = None

    robot = Car()
    driver = Driver()

    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)
    camera.recognitionEnable(timestep)

    # ==============================
    # LIDAR INITIALIZATION
    # ==============================

    # Get the front LiDAR device from the Webots robot.
    # IMPORTANT:
    # The device name must match exactly the name used in the Webots scene tree.
    # If this name does not work, check the robot device list in Webots.
    lidar = robot.getDevice("Sick LMS 291")

    # Enable LiDAR updates using the simulation timestep.
    lidar.enable(timestep)

    print("LiDAR enabled.")
    print(f"LiDAR FOV: {np.degrees(lidar.getFov()):.1f} deg")
    print(f"LiDAR resolution: {lidar.getHorizontalResolution()}")

    display_img = Display("display_image")

    # ==============================
    # SENSOR INITIALIZATION (GPS + Gyro + Distance Sensors)
    # ==============================
    gps = robot.getDevice("gps")
    gyro = robot.getDevice("gyro")
    ds_ml = robot.getDevice("DS_ML")
    ds_mr = robot.getDevice("DS_MR")
    ds_fl = robot.getDevice("DS_FL")
    ds_fr = robot.getDevice("DS_FR")
    ds_rl = robot.getDevice("DS_RL")
    ds_rr = robot.getDevice("DS_RR")

    if gps:
        gps.enable(timestep)
        print("GPS enabled.")
    else:
        print("WARNING: GPS device not found.")

    if gyro:
        gyro.enable(timestep)
        print("Gyro enabled.")
    else:
        print("WARNING: Gyro device not found.")

    if ds_ml:
        ds_ml.enable(timestep)
        print("Distance sensor DS_ML enabled.")
    else:
        print("WARNING: DS_ML device not found.")

    if ds_mr:
        ds_mr.enable(timestep)
        print("Distance sensor DS_MR enabled.")
    else:
        print("WARNING: DS_MR device not found.")

    if ds_fl:
        ds_fl.enable(timestep)
        print("Distance sensor DS_FL enabled.")
    else:
        print("WARNING: DS_FL device not found.")

    if ds_fr:
        ds_fr.enable(timestep)
        print("Distance sensor DS_FR enabled.")
    else:
        print("WARNING: DS_FR device not found.")

    if ds_rl:
        ds_rl.enable(timestep)
        print("Distance sensor DS_RL enabled.")
    else:
        print("WARNING: DS_RL device not found.")

    if ds_rr:
        ds_rr.enable(timestep)
        print("Distance sensor DS_RR enabled.")
    else:
        print("WARNING: DS_RR device not found.")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    joystick = init_ps_controller()

    pid = PIDController(
        kp=KP,
        ki=KI,
        kd=KD,
        output_limit=STEERING_GAIN_LIMIT
    )

    # [CHANGED] Pass the actual camera half-width as error_range so the Error
    # signal is scaled correctly for whatever resolution is set in the Webots
    # scene. Previously hardcoded to 320 (half of 640).
    pid_chart = PIDDebugChart(error_range=camera.getWidth() // 2)

    previous_time = time.time()

    hazard_lights_on = False
    obstacle_status = "CLEAR"
    obstacle_type  = ""        # camera-confirmed type of the blocking object
    vehicle_nearby = False     # True when a vehicle is confirmed < LIDAR_STOP_DISTANCE
    evasion_phase  = 0         # 0=NORMAL, 1=EVADING, 2=RETURNING
    behavior_class = 0         # dataset label — see dataset_mode_v4 header for class map
    _sharp_turn_start_time = None  # timestamp when class 3/4 was entered (None = inactive)
    _evasion_start_time    = None  # set when entering phase 1 — used for safety timeout
    _evasion_clear_time    = None
    _left_ds_triggered     = False # True once a left sensor detects the obstacle alongside
    _saved_gyro_wz         = None  # gyro wZ saved at phase 1 entry — used as heading reference

    # State transition tracking — console prints only fire on change, not every frame
    _prev_evasion_phase  = 0
    _prev_line_detected  = True   # assume line visible at start
    _prev_lidar_stop     = False  # True when last frame triggered emergency stop
    _prev_lidar_ahead    = False  # True when last frame triggered slow-down

    # Previous-frame edge/line states — used by nav_command auto-reset to detect
    # the False→True transition (car exited intersection, lane reference reacquired)
    _prev_edge_detected     = False
    _prev_line_detected_nav = False   # separate from _prev_line_detected used by PID
    _nav_cmd_reset_at       = None    # time.time() when the delay timer started

    # ── Lane side — which lane the car is currently driving in ────────────
    # LANE_RIGHT = right lane, following gravel edge (dataset_mode_v5.LANE_RIGHT)
    # LANE_LEFT  = left lane,  following yellow line (dataset_mode_v5.LANE_LEFT)
    # Toggle with key N. Affects which reference drives the PID and which
    # column is written to the dataset lane_side field.
    lane_side = dataset_mode.LANE_RIGHT   # default: right-lane driving for CIL data

    # ── BC / PID runtime toggles (changed with B and L keys) ──────────────
    # Both start ON → HYBRID mode. Stored as local vars so they don't affect
    # V1.11 if both files share the same Python process (each has its own scope).
    bc_enabled  = BC_ENABLED_DEFAULT
    pid_enabled = PID_ENABLED_DEFAULT

    # EMA state for BC output smoothing — persists across frames
    _bc_smoothed_angle = 0.0

    # Stuck detection timer — set when actual speed drops below threshold in BC mode
    _bc_stuck_start_time = None

    # ── BC model — load once at startup ───────────────────────────────────
    bc_model  = _bc_load_model()
    bc_active = False   # True when BC actually commanded the angle this frame

    # ── CIL model — load once at startup ──────────────────────────────────
    cil_model  = _cil_load_model()
    cil_active = False   # True when CIL commanded the angle this frame
    _cil_smoothed_angle = 0.0

    print("=" * 60)
    print("  Webots Self-Driving Controller — V1.14")
    print("=" * 60)
    print("  Keyboard controls:")
    print("    S          — toggle autonomous mode")
    print("    B          — toggle BC inference (ON/OFF)")
    print("    L          — toggle PID / line-follower (ON/OFF)")
    print("    N          — toggle lane side (LEFT_LANE / RIGHT_LANE)")
    print("    A          — toggle ADAS Monitor window")
    print("    D          — toggle dataset capture mode")
    print("    P          — toggle debug panel")
    print("    C          — save camera image to file")
    print("    Q          — quit")
    print("  PS4:")
    print("    X          — toggle autonomous mode")
    print("    Triangle   — toggle dataset capture mode")
    print("    Square     — save camera image")
    print("    L1         — nav_command = LEFT  (next intersection)")
    print("    R1         — nav_command = RIGHT (next intersection)")
    print("    O (Circle) — nav_command = STRAIGHT (next intersection)")
    print("-" * 55)
    print(f"  BC model  : {'LOADED  →  startup mode = HYBRID (B=ON  L=ON)' if bc_model else 'NOT FOUND  →  PID-only until model is trained'}")
    print(f"  BC speed  : {BC_SPEED_BY_CLASS}")
    print(f"  BC alpha  : {BC_SMOOTH_ALPHA}  clamp: ±{BC_ANGLE_CLAMP} rad")
    print(f"  Stuck thr.: <{BC_STUCK_SPEED_THRESHOLD} km/h for >{BC_STUCK_TIME_THRESHOLD} s → WARNING")
    print(f"  CIL model : {'LOADED  →  L1/R1/O will activate CIL steering' if cil_model else 'NOT FOUND  →  nav_cmd falls back to BC/PID'}")
    print(f"  CIL speed : {CIL_SPEED} km/h  alpha: {CIL_SMOOTH_ALPHA}  clamp: ±{CIL_ANGLE_CLAMP} rad")
    print("=" * 55)

    while robot.step() != -1:
        current_time = time.time()
        dt = current_time - previous_time
        previous_time = current_time

        frame = get_image(camera)

        # ==============================
        # SENSOR READS — GPS + GYRO + DISTANCE SENSORS
        # ==============================
        gps_vals  = gps.getValues()  if gps  else None
        gyro_vals = gyro.getValues() if gyro else None
        ds_ml_val = ds_ml.getValue() if ds_ml else None
        ds_mr_val = ds_mr.getValue() if ds_mr else None
        ds_fl_val = ds_fl.getValue() if ds_fl else None
        ds_fr_val = ds_fr.getValue() if ds_fr else None
        ds_rl_val = ds_rl.getValue() if ds_rl else None
        ds_rr_val = ds_rr.getValue() if ds_rr else None

        # ==============================
        # KEYBOARD CONTROL
        # ==============================

        key = keyboard.getKey()

        if key != -1:
            if key not in last_press or current_time - last_press[key] > DEBOUNCE_TIME:
                last_press[key] = current_time

                if key == keyboard.UP:
                    speed = min(speed + SPEED_INCR, MAX_SPEED)

                elif key == keyboard.DOWN:
                    speed = max(speed - SPEED_INCR, 0)

                elif key == keyboard.RIGHT:
                    angle = min(angle + ANGLE_INCR, MAX_ANGLE)

                elif key == keyboard.LEFT:
                    angle = max(angle - ANGLE_INCR, -MAX_ANGLE)

                elif key in (ord("C"), ord("c")):
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    file_name = current_datetime + ".png"
                    camera.saveImage(os.path.join(os.getcwd(), file_name), 1)
                    print("Image taken")

                elif key == ord("S"):
                    autonomous_mode = not autonomous_mode
                    pid.reset()
                    previous_angle = 0.0
                    line_lost_time = None  # clear any active fallback countdown
                    print(f"Autonomous mode: {autonomous_mode}")

                elif key in (ord("D"), ord("d")):
                    dataset_mode.toggle_dataset()

                elif key in (ord("A"), ord("a")):
                    ADAS_MONITOR_ENABLED = not ADAS_MONITOR_ENABLED
                    if not ADAS_MONITOR_ENABLED:
                        cv2.destroyWindow("ADAS Monitor")
                    print(f"ADAS Monitor: {'ON' if ADAS_MONITOR_ENABLED else 'OFF'}")

                elif key in (ord("P"), ord("p")):
                    DEBUG_PANEL = not DEBUG_PANEL
                    print(f"Debug Panel: {'ON' if DEBUG_PANEL else 'OFF'}")

                elif key in (ord("B"), ord("b")):
                    # Toggle BC inference. Resetting the EMA state prevents
                    # the smoothed angle from jumping when BC is re-enabled.
                    bc_enabled = not bc_enabled
                    if not bc_enabled:
                        _bc_smoothed_angle = 0.0
                        _bc_stuck_start_time = None
                    mode_str = ("HYBRID" if pid_enabled else "FULL BC") if bc_enabled else "PURE PID"
                    print(f"[BC] BC inference: {'ON' if bc_enabled else 'OFF'}  →  mode: {mode_str}")

                elif key in (ord("L"), ord("l")):
                    # Toggle PID / line-follower. Reset the PID integrator when
                    # re-enabling so accumulated error from before doesn't spike.
                    pid_enabled = not pid_enabled
                    if pid_enabled:
                        pid.reset()
                    mode_str = ("HYBRID" if bc_enabled else "PURE PID") if pid_enabled else "FULL BC"
                    print(f"[PID] Line-follower: {'ON' if pid_enabled else 'OFF'}  →  mode: {mode_str}")

                elif key in (ord("N"), ord("n")):
                    # Toggle lane side. Resets PID integrator so accumulated error
                    # from the previous lane reference doesn't carry over.
                    if lane_side == dataset_mode.LANE_RIGHT:
                        lane_side = dataset_mode.LANE_LEFT
                        print("[LANE] Switched to LEFT_LANE  (yellow line reference)")
                    else:
                        lane_side = dataset_mode.LANE_RIGHT
                        print("[LANE] Switched to RIGHT_LANE  (gravel edge reference)")
                    pid.reset()

        # ==============================
        # PS4 CONTROLLER MANUAL MODE
        # ==============================

        if joystick is not None:
            pygame.event.pump()

            if joystick.get_button(BTN_X):
                if "ps_x" not in last_press or current_time - last_press["ps_x"] > 0.5:
                    autonomous_mode = not autonomous_mode
                    pid.reset()
                    previous_angle = 0.0
                    line_lost_time = None  # clear any active fallback countdown
                    last_press["ps_x"] = current_time
                    print(f"Autonomous mode: {autonomous_mode}")

            if joystick.get_button(BTN_TRIANGLE):
                if "ps_triangle" not in last_press or current_time - last_press["ps_triangle"] > 0.5:
                    dataset_mode.toggle_dataset()
                    last_press["ps_triangle"] = current_time

            if joystick.get_button(BTN_SQUARE):
                if "ps_square" not in last_press or current_time - last_press["ps_square"] > 0.5:
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    file_name = current_datetime + ".png"
                    camera.saveImage(os.path.join(os.getcwd(), file_name), 1)
                    print("Image taken from PS controller")
                    last_press["ps_square"] = current_time

            # Navigation command buttons — latch until intersection is cleared.
            # Press before entering the intersection so approach frames are labeled.
            if joystick.get_button(BTN_L1):
                if "ps_l1" not in last_press or current_time - last_press["ps_l1"] > 0.5:
                    dataset_mode.set_nav_command(dataset_mode.CMD_LEFT)
                    last_press["ps_l1"] = current_time

            if joystick.get_button(BTN_R1):
                if "ps_r1" not in last_press or current_time - last_press["ps_r1"] > 0.5:
                    dataset_mode.set_nav_command(dataset_mode.CMD_RIGHT)
                    last_press["ps_r1"] = current_time

            if joystick.get_button(BTN_CIRCLE):
                if "ps_circle" not in last_press or current_time - last_press["ps_circle"] > 0.5:
                    dataset_mode.set_nav_command(dataset_mode.CMD_STRAIGHT)
                    last_press["ps_circle"] = current_time

            # Manual joystick only when autonomous is OFF
            if not autonomous_mode:
                left_stick_x = apply_deadzone(joystick.get_axis(0))
                angle = left_stick_x * MAX_ANGLE

                l2 = joystick.get_axis(2)
                r2 = joystick.get_axis(5)

                # PS4 triggers report -1.0 (released) to +1.0 (fully pressed);
                # shift and halve to map to the [0, 1] range expected below.
                throttle = (r2 + 1) / 2
                brake = (l2 + 1) / 2

                speed = throttle * CONTROLLER_MAX_SPEED - brake * abs(REVERSE_MAX_SPEED)

        # ==============================
        # LANE REFERENCE DETECTION — yellow line (left) + gravel edge (right)
        # Both run every frame so the debug windows and dataset labels are
        # always up to date, regardless of which lane is active.
        # ==============================

        line_detected, line_x, desired_x, debug_frame, yellow_roi = detect_yellow_line(frame)
        edge_detected, edge_x, edge_error, edge_mask = detect_gravel_edge(frame)

        # Nav-command auto-reset with post-intersection hold.
        # Step 1 — on the False→True transition of the lane reference start the timer.
        # Step 2 — reset only after NAV_CMD_RESET_DELAY seconds have elapsed.
        # This keeps the command (and 4 Hz capture rate) active through the approach,
        # the intersection itself, and a few seconds of the post-turn straight.
        if dataset_mode.get_nav_command() != dataset_mode.CMD_LANE_FOLLOW:
            reacquired = False
            if lane_side == dataset_mode.LANE_RIGHT:
                reacquired = edge_detected and not _prev_edge_detected
            else:
                reacquired = line_detected and not _prev_line_detected_nav

            if reacquired and _nav_cmd_reset_at is None:
                _nav_cmd_reset_at = current_time
                print(f"[NAV] Lane reference reacquired — resetting nav_command "
                      f"in {NAV_CMD_RESET_DELAY:.0f} s")

            if _nav_cmd_reset_at is not None and (current_time - _nav_cmd_reset_at) >= NAV_CMD_RESET_DELAY:
                dataset_mode.reset_nav_command()
                _nav_cmd_reset_at = None
                print("[NAV] nav_command → LANE_FOLLOW")

        else:
            _nav_cmd_reset_at = None   # clear timer if command was reset externally

        _prev_edge_detected     = edge_detected
        _prev_line_detected_nav = line_detected

        # ==============================
        # LIDAR FRONT OBSTACLE DETECTION
        # ==============================

        # Default LiDAR values for this frame.
        lidar_obstacle = False
        lidar_obstacle_angle = None
        lidar_obstacle_distance = None

        if ENABLE_LIDAR_OBSTACLE_DETECTION:

            # Process only the front-center LiDAR sector.
            lidar_obstacle, lidar_obstacle_angle, lidar_obstacle_distance = process_lidar_data(lidar)
                
                

            # Draw LiDAR status on the debug frame.
            if lidar_obstacle:
                cv2.putText(
                    debug_frame,
                    f"LIDAR obstacle: {lidar_obstacle_distance:.1f} m | angle: {lidar_obstacle_angle} deg",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    debug_frame,
                    "LIDAR: clear",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # ==========================================
        # CAMERA RECOGNITION (built-in neural network)
        # Filtered to relevant driving objects only
        # ==========================================
        
        recognized_objects = []

        # Object categories for color coding
        COLOR_VEHICLE = (0, 0, 255)       # Red — vehicles (bus, car)
        COLOR_TRAFFIC_LIGHT = (0, 255, 255)  # Yellow
        COLOR_SIGN = (255, 255, 0)        # Cyan
        COLOR_OBSTACLE = (255, 0, 0)      # Blue — TrafficCone, Barrel
        
        # Filtered categories (ignore buildings, trees, roads, etc.)
        CATEGORIES = {
            "vehicle": ["bus", "car", "truck", "van", "toyota", "lincoln", "citroen", "bmw", "mercedes", "suv"],
            "traffic_light": ["traffic light"],
            "sign": ["caution panel", "order panel", "speed limit panel", "yield panel", "highway sign"],
            "obstacle": ["traffic cone", "barrel", "crash barrier"]
        }
        
        num_obj = camera.getRecognitionNumberOfObjects()
        if num_obj > 0 and ENABLE_OBJECT_DETECTION:
            objects = camera.getRecognitionObjects()
            for obj in objects:
                model = obj.getModel()
                if DEBUG_PRINT:
                    m = model.lower()
                    if any(c in m for c in CATEGORIES["sign"]):
                        print(f"[SIGN MATCH] '{model}'")
                    else:
                        for sign_cat in CATEGORIES["sign"]:
                            if sign_cat in m:
                                print(f"[SIGN FOUND IN MODEL] model='{model}' with '{sign_cat}'")
                                break
                        else:
                            print(f"[SIGN NO MATCH] model='{model}' (len={len(model)})")
                    if any(c in m for c in CATEGORIES["vehicle"]):
                        print(f"[VEHICLE MATCH] {model}")
            
            for obj in objects:
                model = obj.getModel()
                pos_img = obj.getPositionOnImage()
                size_img = obj.getSizeOnImage()
                x, y = pos_img
                w, h = size_img
                
                # Skip if out of frame bounds
                if x + w < 0 or y + h < 0:
                    continue
                
                # Distance from camera (z-axis)
                obj_pos = obj.getPosition()
                distance = obj_pos[2] if len(obj_pos) > 2 else 0
                
                # Classify into category
                obj_type = None
                color = None
                display_name = model
                
                model_lower = model.lower()
                if any(c in model_lower for c in CATEGORIES["vehicle"]):
                    if "bmw" in model_lower:
                        continue  # Skip our own car
                    obj_type = "vehicle"
                    color = COLOR_VEHICLE
                elif any(c in model_lower for c in CATEGORIES["traffic_light"]):
                    obj_type = "traffic_light"
                    color = COLOR_TRAFFIC_LIGHT
                elif any(c in model_lower for c in CATEGORIES["sign"]):
                    obj_type = "sign"
                    color = COLOR_SIGN
                    # Extract just the sign type name
                    for sign_type in CATEGORIES["sign"]:
                        if sign_type in model_lower:
                            display_name = sign_type
                            break
                elif any(c in model_lower for c in CATEGORIES["obstacle"]):
                    obj_type = "obstacle"
                    color = COLOR_OBSTACLE
                else:
                    continue  # Skip buildings, trees, etc.
                
                recognized_objects.append({"type": obj_type, "distance": distance, "model": display_name})
                
                # Draw bounding box
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(debug_frame, display_name, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)   

        # ── PID error source: yellow line (left lane) or gravel edge (right lane) ──
        # lane_side determines which reference drives the PID controller.
        # In both cases, a positive error means the car is drifted left of target;
        # negative error means drifted right. The PID output and direction convention
        # are identical — only the source of the error measurement changes.

        if lane_side == dataset_mode.LANE_RIGHT:
            # Right lane: gravel edge drives the PID.
            # Debug overlay mirrors the yellow line interface:
            #   - Contour lines  (≈ raw Hough lines)  — light green, thin
            #   - Thick green line at edge_x           (≈ yellow reference line)
            #   - Magenta line at target_x             (≈ virtual lane center)
            #   - Blue dot at y_ref                    (≈ PID reference point)
            if edge_detected:
                error = edge_error

                h_fr     = frame.shape[0]
                w_fr     = frame.shape[1]
                y_ref    = int(h_fr * Y_REF_RATIO)
                y_roi    = int(h_fr * EDGE_CROP_TOP)   # ROI offset in full frame
                target_x = int(w_fr * EDGE_TARGET_X_RATIO)

                # Contour lines of the gravel mask — projected back to full frame
                # coordinates by adding y_roi offset (equivalent to Hough raw lines)
                contours, _ = cv2.findContours(
                    edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_shifted = cnt + np.array([[[0, y_roi]]])
                    cv2.drawContours(debug_frame, [cnt_shifted], -1, (0, 180, 80), 1)

                # Thick green vertical line at detected edge_x
                # (equivalent to the thick cyan/yellow reference line)
                cv2.line(debug_frame,
                         (edge_x, int(h_fr * 0.45)),
                         (edge_x, h_fr),
                         (0, 255, 80), 8)

                # Magenta line at target_x — where we want the edge to be
                # (equivalent to the magenta virtual lane center line)
                cv2.line(debug_frame,
                         (target_x, int(h_fr * 0.45)),
                         (target_x, h_fr),
                         (255, 0, 255), 3)

                # Blue dot at PID reference point — same as yellow line
                cv2.circle(debug_frame, (edge_x, y_ref), 9, (255, 0, 0), -1)

                cv2.putText(debug_frame,
                            f"Edge X:{edge_x}  Target:{target_x}  Error:{error:.0f}  [RIGHT LANE]",
                            (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
            else:
                error = 0
                cv2.putText(debug_frame,
                            "Gravel edge: NOT DETECTED  [RIGHT LANE]",
                            (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)

        elif line_detected:
            # Left lane: virtual lane center from yellow line + offset.
            lane_center_x   = line_x + LANE_OFFSET_PX
            lane_center_x   = max(0, min(frame.shape[1] - 1, lane_center_x))
            camera_center_x = frame.shape[1] // 2
            error = lane_center_x - camera_center_x

            # desired_x center reference (was inside detect_yellow_line — moved here
            # so it only draws in left-lane mode, not when gravel edge is active)
            cv2.line(debug_frame,
                     (desired_x, int(frame.shape[0] * 0.45)),
                     (desired_x, frame.shape[0]),
                     (255, 0, 255), 3)
            cv2.putText(debug_frame,
                        f"Line X:{line_x}  Lane Center:{lane_center_x}  Error:{error}",
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            error = 0

            cv2.putText(
                debug_frame,
                "Target: keep car centered in right lane using yellow line",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )


        # ==============================
        # OBSTACLE CLASSIFICATION (all modes — display + dataset)
        # ==============================
        # Runs every frame so obstacle_type, vehicle_nearby, and evasion_phase
        # are populated during manual recording too.
        obstacle_type = ""
        vehicle_nearby = False

        if lidar_obstacle and lidar_obstacle_distance is not None:
            if lidar_obstacle_distance < LIDAR_STOP_DISTANCE:
                nearby = [o for o in recognized_objects if abs(o["distance"]) < LIDAR_STOP_DISTANCE]
                vehicle_nearby = any(o["type"] == "vehicle" for o in nearby)
                if vehicle_nearby:
                    obstacle_type = "VEHICLE"
                    obstacle_status = "VEHICLE"
                elif any(o["type"] == "obstacle" for o in nearby):
                    obstacle_type = "OBSTACLE"
                    obstacle_status = "OBSTACLE"
                elif nearby:
                    obstacle_type = nearby[0]["type"].upper()
                    obstacle_status = obstacle_type
                else:
                    obstacle_type = "LIDAR_ONLY"
                    obstacle_status = "LIDAR_ONLY"
            elif lidar_obstacle_distance < LIDAR_MAX_DISTANCE:
                obstacle_type = "OBJECT_AHEAD"
                obstacle_status = "OBJECT AHEAD"
        else:
            obstacle_status = "CLEAR"

        # Evasion phase state machine
        # DS latch + clear check runs first — physical clearance overrides vehicle_nearby.
        # (camera may still see the vehicle from behind even after the car has fully passed it)
        if evasion_phase == 1:
            # Update latch every frame while evading
            if (ds_fl_val is not None and ds_fl_val <= DS_LEFT_CLEAR_THRESHOLD) or \
               (ds_ml_val is not None and ds_ml_val <= DS_LEFT_CLEAR_THRESHOLD) or \
               (ds_rl_val is not None and ds_rl_val <= DS_LEFT_CLEAR_THRESHOLD):
                _left_ds_triggered = True

            left_clear = (
                (ds_fl_val is None or ds_fl_val > DS_LEFT_CLEAR_THRESHOLD) and
                (ds_ml_val is None or ds_ml_val > DS_LEFT_CLEAR_THRESHOLD) and
                (ds_rl_val is None or ds_rl_val > DS_LEFT_CLEAR_THRESHOLD)
            )
            timed_out = (_evasion_start_time is not None and
                         (current_time - _evasion_start_time) > EVADING_MAX_DURATION)

            if (_left_ds_triggered and left_clear) or timed_out:
                # Obstacle physically passed — go to RETURNING regardless of camera/LiDAR
                evasion_phase = 2
                _evasion_clear_time = current_time
                reason = "left sensors clear" if _left_ds_triggered else f"timeout {EVADING_MAX_DURATION:.0f}s"
                print(f"[STATE] EVADING → RETURNING  ({reason}, DS_MR steering,"
                      f" timeout={RETURNING_DURATION:.0f}s)")
            elif vehicle_nearby:
                evasion_phase = 1  # still alongside — keep evading
            # else: vehicle_nearby dropped but latch not yet set → keep evading, wait for DS

        elif vehicle_nearby:
            # Fresh entry into EVADING from NORMAL or RETURNING
            if _prev_evasion_phase != 1:
                _evasion_start_time = current_time
                _left_ds_triggered = False  # reset for this new evasion event
                _saved_gyro_wz = gyro_vals[2] if gyro_vals is not None else None
                wz_str = f"{_saved_gyro_wz:.3f}" if _saved_gyro_wz is not None else "N/A"
                print(f"[STATE] {'NORMAL' if _prev_evasion_phase == 0 else 'RETURNING'}"
                      f" → EVADING  (vehicle < {LIDAR_STOP_DISTANCE:.0f}m,"
                      f" type={obstacle_type}, wZ_ref={wz_str})")
            evasion_phase = 1
            _evasion_clear_time = None
        elif evasion_phase == 2 and _evasion_clear_time is not None:
            wz_now = gyro_vals[2] if gyro_vals is not None else None
            wz_aligned = (_saved_gyro_wz is not None and wz_now is not None and
                          abs(wz_now - _saved_gyro_wz) <= WZ_RETURN_TOLERANCE)
            timed_out_p2 = (current_time - _evasion_clear_time) > RETURNING_DURATION

            if line_detected or wz_aligned or timed_out_p2:
                evasion_phase = 0
                _evasion_clear_time = None
                if line_detected:
                    print("[STATE] RETURNING → NORMAL  (yellow line found — PID resumes)")
                elif wz_aligned:
                    print(f"[STATE] RETURNING → NORMAL  (wZ aligned:"
                          f" cur={wz_now:.3f}  ref={_saved_gyro_wz:.3f})")
                else:
                    print(f"[STATE] RETURNING → NORMAL  (timeout {RETURNING_DURATION:.0f}s)")
                _saved_gyro_wz = None
        else:
            evasion_phase = 0
        _prev_evasion_phase = evasion_phase

        # ==============================
        # BEHAVIOR CLASS — dataset label (computed every frame)
        # Priority: 1/2 (vehicle evasion) > 3/4 (intersection turns) > 0 (normal)
        # Debounce: hysteresis on angle threshold + minimum dwell frame counter.
        # ==============================
        if evasion_phase == 1:
            behavior_class = 1
            _sharp_turn_start_time = None
        elif evasion_phase == 2:
            behavior_class = 2
            _sharp_turn_start_time = None
        elif (_sharp_turn_start_time is not None and
              (current_time - _sharp_turn_start_time) < SHARP_TURN_DWELL_SECS):
            pass  # hold current class 3/4 for the full dwell window
        elif not line_detected and evasion_phase == 0 and angle < -SHARP_TURN_ENTRY_ANGLE:
            behavior_class = 3  # SHARP_TURN_LEFT
            _sharp_turn_start_time = current_time
        elif not line_detected and evasion_phase == 0 and angle > SHARP_TURN_ENTRY_ANGLE:
            behavior_class = 4  # SHARP_TURN_RIGHT
            _sharp_turn_start_time = current_time
        else:
            behavior_class = 0
            _sharp_turn_start_time = None

        # Human-readable names for behavior_class — used in HUD and console prints
        phase_labels = {0: "NORMAL", 1: "EVADING", 2: "RETURNING",
                        3: "SHARP_LEFT", 4: "SHARP_RIGHT"}

        # ==============================
        # AUTONOMOUS MODE — BC / PID ARBITRATION + LIDAR SAFETY
        # ==============================
        # Priority order (highest to lowest):
        #   1. Evasion phases 1/2    — vehicle alongside or just passed
        #   2. LIDAR emergency stop  — obstacle < LIDAR_STOP_DISTANCE
        #   3. LIDAR slow-down       — obstacle between stop and max distance
        #   4. Normal driving        — four-mode BC/PID arbitration
        # ==============================

        if autonomous_mode:

            bc_active = False  # True when BC commanded the angle this frame

            # ── Phase 1: EVADING ──────────────────────────────────────────────
            # A vehicle is alongside or approaching. PID cannot help — yellow
            # line is not visible from the evasion lane.
            #
            # Sub-phase A (_left_ds_triggered=False):
            #   BC steers right to pull out from behind the vehicle.
            # Sub-phase B (_left_ds_triggered=True):
            #   DS_ML sensor holds parallel distance until the vehicle is fully
            #   passed. BC is not called here — sensor feedback is more precise
            #   for lateral positioning when adjacent to another vehicle.
            if evasion_phase == 1:
                bc_active = True
                hazard_lights_on = vehicle_nearby
                speed = int(0.35 * AUTONOMOUS_SPEED)
                brake = 0.0

                if not _left_ds_triggered:
                    # Sub-phase A: BC steers the car rightward past the obstacle.
                    # NOTE: bc_enabled is intentionally NOT checked here.
                    # Evasion always uses BC when the model is loaded, regardless
                    # of the B-key toggle. The toggle only affects normal driving.
                    # A vehicle alongside is safety-critical — BC gives the best
                    # visual steering; falling back to fixed angle risks collision.
                    if bc_model is not None:
                        try:
                            _bc_smoothed_angle = _bc_predict_smooth(bc_model, frame, _bc_smoothed_angle)
                            angle = _bc_smoothed_angle
                            previous_angle = angle
                        except Exception as _e:
                            print(f"[BC] Inference error in EVADING: {_e} — holding angle")
                            angle = previous_angle
                    else:
                        # BC model not loaded — hold the last angle to maintain trajectory
                        angle = previous_angle
                    cv2.putText(debug_frame,
                                f"AUTO | BC EVADING  angle:{angle:.3f}",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 165, 255), 2)
                else:
                    # Sub-phase B: DS_ML proportional control keeps lateral gap.
                    steer = 0.0  # default: straight ahead
                    if ds_ml_val is not None:
                        if ds_ml_val < DS_LEFT_PARALLEL_MIN:
                            # Too close to the passing vehicle — nudge right
                            steer = PARALLEL_NUDGE_ANGLE
                        elif ds_ml_val < DS_LEFT_PARALLEL_TARGET:
                            # Proportional: the closer we are, the more we steer right
                            ratio = 1.0 - (ds_ml_val - DS_LEFT_PARALLEL_MIN) / \
                                          (DS_LEFT_PARALLEL_TARGET - DS_LEFT_PARALLEL_MIN)
                            steer = PARALLEL_NUDGE_ANGLE * ratio
                        # else: gap is healthy — go straight (steer stays 0.0)
                    angle = 0.6 * previous_angle + 0.4 * steer
                    angle = max(-MAX_ANGLE, min(MAX_ANGLE, angle))
                    previous_angle = angle
                    ml_str = f"{ds_ml_val:.0f}" if ds_ml_val is not None else "N/A"
                    cv2.putText(debug_frame,
                                f"AUTO | PARALLEL  angle:{angle:.3f}  DS_ML:{ml_str}",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 200, 100), 2)

            # ── Phase 2: RETURNING ────────────────────────────────────────────
            # Vehicle has passed. Car is in the right lane; no yellow line visible.
            # DS_MR keeps the car away from the right road guard while a base
            # left steer gradually migrates the car back toward the left lane.
            # Exits when line_detected, gyro wZ re-aligns, or timeout fires.
            elif evasion_phase == 2:
                bc_active = True   # flag tells the LIDAR/PID blocks to skip
                hazard_lights_on = False
                brake = 0.0
                speed = int(RETURN_SPEED_RATIO * AUTONOMOUS_SPEED)

                steer = RETURN_LANE_ANGLE  # base: gentle left

                if ds_mr_val is not None:
                    if ds_mr_val < DS_RETURN_RIGHT_MIN:
                        # Right guard is too close — reduce left steer for safety
                        steer = max(steer, -0.04)
                    elif ds_mr_val < DS_RETURN_RIGHT_TARGET:
                        # Proportional: closer to guard = softer left steer
                        ratio = ds_mr_val / DS_RETURN_RIGHT_TARGET
                        steer = RETURN_LANE_ANGLE * ratio

                angle = 0.6 * previous_angle + 0.4 * steer
                angle = max(-MAX_ANGLE, min(MAX_ANGLE, angle))
                previous_angle = angle

                ds_mr_str = f"{ds_mr_val:.0f}" if ds_mr_val is not None else "N/A"
                cv2.putText(debug_frame,
                            f"AUTO | RETURNING  angle:{angle:.3f}  DS_MR:{ds_mr_str}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

            # ── LIDAR safety override (only when not in evasion phases) ───────
            # Phases 1/2 above manage their own speed and braking. This block
            # only runs for phase 0 (normal / sharp turn situations).
            elif lidar_obstacle and lidar_obstacle_distance < LIDAR_MAX_DISTANCE:

                brake = 0.0

                if lidar_obstacle_distance < LIDAR_STOP_DISTANCE:
                    # Inside emergency stop zone — full brake regardless of BC/PID state.
                    # obstacle_type and vehicle_nearby are already set by the
                    # classification block earlier in the loop.
                    if not _prev_lidar_stop:
                        print(f"[STATE] → EMERGENCY STOP  (obstacle={obstacle_type},"
                              f" dist={lidar_obstacle_distance:.1f}m, hazard={vehicle_nearby})")
                    _prev_lidar_stop  = True
                    _prev_lidar_ahead = False
                    speed  = 0
                    angle  = 0.0
                    previous_angle = 0.0
                    brake  = 1.0
                    hazard_lights_on = vehicle_nearby

                else:
                    # Object is between stop distance and max range — slow down but
                    # keep whichever steering controller is active (BC or PID).
                    if not _prev_lidar_ahead:
                        print(f"[STATE] → OBJECT AHEAD  (dist={lidar_obstacle_distance:.1f}m,"
                              f" speed reduced to 50%)")
                    _prev_lidar_stop  = False
                    _prev_lidar_ahead = True
                    hazard_lights_on  = False
                    obstacle_status   = "OBJECT AHEAD"

                    if line_detected:
                        pid_output = pid.compute(error, dt)
                        angle = 0.6 * previous_angle + 0.4 * pid_output
                        previous_angle = angle
                    else:
                        angle = previous_angle * 0.5  # gentle fade when line is lost

                    if abs(angle) < 0.03:
                        angle = 0.0
                    previous_angle = angle
                    speed = 0.5 * AUTONOMOUS_SPEED
                    brake = 0.0

            # ── Normal driving — CIL intercept then BC/PID arbitration ──────
            # Reached only when evasion_phase=0 and no LIDAR emergency.
            else:
                _prev_lidar_stop  = False
                _prev_lidar_ahead = False
                hazard_lights_on  = False
                brake = 0

                _nav_cmd = dataset_mode.get_nav_command()

                # ── CIL MODE — human set a nav_command via L1/R1/O ────────────
                # Bypasses BC/PID entirely while the command is active.
                # Falls back to BC/PID if the model file was not found.
                if _nav_cmd != dataset_mode.CMD_LANE_FOLLOW and cil_model is not None:
                    cil_active = True
                    bc_active  = False
                    brake      = 0.0
                    speed      = CIL_SPEED
                    try:
                        _cil_smoothed_angle = _cil_predict_smooth(
                            cil_model, frame, _nav_cmd, _cil_smoothed_angle)
                        angle          = _cil_smoothed_angle
                        previous_angle = angle
                        _cmd_label = _CIL_CMD_NAMES.get(_nav_cmd, "?")
                        cv2.putText(debug_frame,
                                    f"AUTO | CIL {_cmd_label}  angle:{angle:.3f}",
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 200, 0), 2)
                    except Exception as _e:
                        print(f"[CIL] Inference error: {_e} — holding angle={previous_angle:.3f}")
                        angle      = previous_angle
                        cil_active = False

                else:
                    # nav_cmd == LANE_FOLLOW (or CIL model missing) → BC/PID
                    cil_active          = False
                    _cil_smoothed_angle = 0.0   # reset EMA so it starts fresh next activation

                # ── MODE A: HYBRID — PID handles class 0, BC handles 1/2/3/4 ──
                # Skipped when CIL is active (cil_active=True short-circuits below).
                if not cil_active and bc_enabled and bc_model is not None and pid_enabled \
                        and line_detected and behavior_class == 0:

                    if not _prev_line_detected:
                        print("[STATE] LINE_LOST → NORMAL  (yellow line reacquired — PID resumes)")
                    line_lost_time     = None
                    _prev_line_detected = True
                    bc_active          = False

                    pid_output = pid.compute(error, dt)
                    angle  = 0.6 * previous_angle + 0.4 * pid_output
                    previous_angle = angle
                    speed  = AUTONOMOUS_SPEED - min(abs(error) * 0.15, 10)
                    speed  = max(speed, 8)
                    cv2.putText(debug_frame, f"AUTO | HYBRID — PID  angle:{angle:.3f}",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # ── MODE B: BC active (covers HYBRID non-class-0, and FULL BC) ──
                elif not cil_active and bc_enabled and bc_model is not None:
                    bc_active    = True
                    line_lost_time = None   # BC handles missing line — no countdown

                    try:
                        _bc_smoothed_angle = _bc_predict_smooth(bc_model, frame, _bc_smoothed_angle)
                        angle  = _bc_smoothed_angle
                        previous_angle = angle
                        speed  = BC_SPEED_BY_CLASS.get(behavior_class, 30.0)
                        brake  = 0.0
                        _prev_line_detected = line_detected
                        cv2.putText(debug_frame,
                                    f"AUTO | BC {phase_labels.get(behavior_class,'?'):10s}  angle:{angle:.3f}",
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    except Exception as _e:
                        # Inference failure — hold last angle and let the loop continue
                        print(f"[BC] Inference error: {_e} — holding angle={previous_angle:.3f}")
                        angle     = previous_angle
                        bc_active = False

                # ── MODE C: PURE PID (bc_enabled=False) ──────────────────────
                elif not cil_active and pid_enabled:
                    bc_active = False

                    if line_detected:
                        if not _prev_line_detected:
                            print("[STATE] LINE_LOST → NORMAL  (yellow line reacquired — PID resumes)")
                        line_lost_time      = None
                        _prev_line_detected = True

                        pid_output = pid.compute(error, dt)
                        angle  = 0.6 * previous_angle + 0.4 * pid_output
                        previous_angle = angle
                        speed  = AUTONOMOUS_SPEED - min(abs(error) * 0.15, 10)
                        speed  = max(speed, 8)
                        cv2.putText(debug_frame, f"AUTO | PID  angle:{angle:.3f}",
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    else:
                        # Line lost with PID only — decay fallback + 10 s countdown
                        if line_lost_time is None:
                            line_lost_time = current_time
                            print("[STATE] NORMAL → LINE_LOST  (DECAY fallback, countdown=10s)")
                        _prev_line_detected = False

                        time_remaining = max(0.0, 10.0 - (current_time - line_lost_time))
                        countdown      = int(np.ceil(time_remaining))

                        angle = previous_angle * 0.5
                        speed = 10
                        if abs(angle) < 0.03:
                            angle = 0.0

                        cv2.putText(debug_frame, "AUTO | PID — Searching yellow line",
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        countdown_color = (0, 100, 255) if countdown > 3 else (0, 0, 255)
                        cv2.putText(debug_frame, f"Line lost — auto MANUAL in: {countdown}s",
                                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, countdown_color, 2)

                        if time_remaining <= 0:
                            autonomous_mode = False
                            line_lost_time  = None
                            pid.reset()
                            previous_angle  = 0.0
                            angle = 0.0
                            print("[STATE] Line lost 10 s — reverted to MANUAL mode automatically")

                # ── MODE D: SAFE HOLD — both BC and PID disabled ─────────────
                elif not cil_active:
                    bc_active = False
                    angle = 0.0
                    speed = 0.0
                    brake = 0.3
                    cv2.putText(debug_frame, "SAFE HOLD — BC and PID both disabled  [press S]",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # ── BC stuck detection ────────────────────────────────────────
                # Reads the actual vehicle speed (not the commanded setpoint) to
                # detect when the model has steered the car into a wall or stopped
                # at an unexpected obstacle. Warning fires every BC_STUCK_TIME_THRESHOLD
                # seconds so the operator knows to intervene.
                if bc_active:
                    try:
                        actual_speed = abs(driver.getCurrentSpeed())
                    except Exception:
                        actual_speed = speed  # fallback if API unavailable

                    if actual_speed < BC_STUCK_SPEED_THRESHOLD and speed > BC_STUCK_SPEED_THRESHOLD:
                        # Car is commanded to move but is not moving
                        if _bc_stuck_start_time is None:
                            _bc_stuck_start_time = current_time
                        elif (current_time - _bc_stuck_start_time) > BC_STUCK_TIME_THRESHOLD:
                            print(
                                f"[BC] WARNING: Vehicle appears stuck! "
                                f"Commanded={speed:.0f} km/h  Actual={actual_speed:.1f} km/h  "
                                f"for >{BC_STUCK_TIME_THRESHOLD:.0f}s  "
                                f"(class={behavior_class} {phase_labels.get(behavior_class,'?')}). "
                                f"Press S to exit autonomous mode and reposition."
                            )
                            _bc_stuck_start_time = current_time  # reset so warning repeats
                    else:
                        _bc_stuck_start_time = None  # moving normally — clear timer
                else:
                    _bc_stuck_start_time = None  # BC not in control — no stuck check needed

        else:
            cv2.putText(
                debug_frame,
                "MANUAL MODE - vision debug active",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # ==========================================
        # HAZARD LIGHT AND OBSTACLE STATUS DISPLAY
        # Drawn after autonomous-mode logic so hazard_lights_on and
        # obstacle_status reflect the current frame's decision.
        # ==========================================

        cv2.putText(
            debug_frame,
            f"Hazard Lights: {'ON' if hazard_lights_on else 'OFF'}",
            (30, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255) if hazard_lights_on else (0, 255, 0),
            2
        )

        phase_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255),
                        3: (255, 200, 0), 4: (255, 200, 0)}
        cv2.putText(
            debug_frame,
            f"Obstacle: {obstacle_status}  Type: {obstacle_type or '--'}",
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )
        cv2.putText(
            debug_frame,
            f"Behavior: {behavior_class} ({phase_labels.get(behavior_class, '?')})",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            phase_colors.get(behavior_class, (210, 210, 210)),
            2
        )

        # Dataset mode indicator
        if dataset_mode.dataset_mode:
            cv2.putText(
                debug_frame,
                f"[DS] CAPTURE #{dataset_mode.dataset_counter}  Mode: {'ON' if dataset_mode.dataset_mode else 'OFF'}",
                (30, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Debug Panel indicator
        cv2.putText(
            debug_frame,
            f"DEBUG PANEL: {'ON' if DEBUG_PANEL else 'OFF'} [P to toggle]",
            (30, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

        # Speed overlay — drawn on every frame in both manual and autonomous
        # modes so the operator always has current speed visible in the main
        # debug window without needing to read the separate PID chart.
        # Color matches the active mode: cyan = autonomous, green = manual.
        # Y position is anchored to the bottom of the frame using shape[0] so
        # it stays in the right place regardless of camera resolution.
        speed_color = (0, 255, 255) if autonomous_mode else (0, 255, 0)
        cv2.putText(
            debug_frame,
            f"Speed: {speed:.0f} km/h",
            (30, debug_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            speed_color,
            2
        )

        display_frame = cv2.resize(debug_frame, (640, 300))

        # ==============================
        # DATASET MODE CAPTURE
        # ==============================
        # Placed after all control decisions so angle/speed/brake/line_detected
        # reflect what was actually applied to the car this frame.
        dataset_mode.try_capture(
            frame, angle, speed, brake, current_time,
            autonomous_mode=autonomous_mode,
            lane_side=lane_side,
            line_detected=line_detected,
            edge_detected=edge_detected,
            gps_vals=gps_vals,
            lidar_obstacle_detected=lidar_obstacle,
            lidar_obstacle_distance=lidar_obstacle_distance,
            behavior_class=behavior_class,
            obstacle_type=obstacle_type,
        )

        # ==========================================
        # HAZARD LIGHT CONTROL
        # ==========================================

        driver.setHazardFlashers(hazard_lights_on)

        # ==============================
        # SEND COMMANDS TO CAR
        # ==============================

        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)
        driver.setBrakeIntensity(brake)

        pid_chart.update(error, angle, speed)
        pid_chart.show()

        # ==============================
        # ADAS MONITOR
        # ==============================
        if ADAS_MONITOR_ENABLED:
            draw_adas_monitor(speed, angle, brake, gps_vals, gyro_vals,
                              ds_ml_val, ds_mr_val, ds_fl_val, ds_fr_val, ds_rl_val, ds_rr_val,
                              autonomous_mode=autonomous_mode, evasion_phase=behavior_class,
                              line_detected=line_detected, bc_active=bc_active,
                              lidar_stop=_prev_lidar_stop, lidar_ahead=_prev_lidar_ahead,
                              lidar_obstacle=lidar_obstacle, lidar_distance=lidar_obstacle_distance,
                              lidar_angle=lidar_obstacle_angle, obstacle_type=obstacle_type,
                              vehicle_nearby=vehicle_nearby,
                              parallel_active=(evasion_phase == 1 and _left_ds_triggered),
                              saved_gyro_wz=_saved_gyro_wz,
                              current_gyro_wz=(gyro_vals[2] if gyro_vals is not None else None),
                              bc_enabled=bc_enabled, pid_enabled=pid_enabled,
                              lane_side=lane_side, edge_detected=edge_detected,
                              nav_command=dataset_mode.get_nav_command(),
                              cil_active=cil_active,
                              cil_model_loaded=(cil_model is not None))

        # Display image in Webots display
        if DEBUG_PANEL:
            display_image(display_img, display_frame)

            # Optional OpenCV debug window
            cv2.imshow("Self Driving Debug", display_frame)

            # Mask debug window swaps content based on active lane reference.
            # RIGHT lane → gravel edge mask (tune EDGE_BRIGHT_THRESH here).
            # LEFT lane  → yellow line mask  (original behaviour).
            if lane_side == dataset_mode.LANE_RIGHT:
                edge_mask_bgr = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Gravel Edge Debug", cv2.resize(edge_mask_bgr, (640, 300)))
                try: cv2.destroyWindow("Yellow Mask Debug")
                except Exception: pass
            else:
                cv2.imshow("Yellow Mask Debug", cv2.resize(yellow_roi, (640, 300)))
                try: cv2.destroyWindow("Gravel Edge Debug")
                except Exception: pass

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Only check for quit when debug is off
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



    ## in order to create the connection to the Webot ,, need to run this in the terminal shell.
    #(venv) PS D:\ML\Projects\Project_5_MR4010.10_Navegacion\MR4010.10_Navegacion> & "C:\Program Files\Webots\msys64\mingw64\bin\webots-controller.exe" .\src\simple_controller_act_2_1_V1.5.py --stdout-redirect
    #to enable the keyboard in webot you need to click the webot window

    #Left stick left/right = steering
    #R2 = throttle
    #L2 = brake / reverse
    #X = toggle autonomous mode
    #Square = take image
    #Keyboard still works in parallel