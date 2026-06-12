# ============================================================
# V1.9.12 — Webots Self-Driving Controller
# ============================================================
#
# LANE FOLLOWING
#   - Grayscale brightness threshold (120–200) replaces HSV yellow mask.
#   - Narrow trapezoidal ROI + morphological open/close/dilate before Canny+Hough.
#   - Weighted-average Hough line fit with slope and boundary guards.
#   - Moving-average smoothing (deque of 10 frames) + exponential temporal filter.
#   - Virtual lane center estimated as yellow-line X + 110 px offset (right lane).
#   - PID controller (KP=0.0015, KI=0.000, KD=0.0003) with output clamped to ±0.7 rad.
#   - Steering smoothed per frame: 60 % previous + 40 % PID output.
#   - Speed reduced proportionally with lateral error; floor at 8 km/h.
#   - Line-lost fallback: hold last steering (×0.5 decay) at 10 km/h for 10 s,
#     then revert automatically to MANUAL mode.
#
# LIDAR OBSTACLE DETECTION  (Sick LMS 291)
#   - Front-center 40° inspection window (±20 samples from center).
#   - Max detection range: 25 m; emergency-stop threshold: 15 m.
#   - Between 15–25 m: slow to 50 % speed, lane following stays active.
#   - Below 15 m: full brake, zero steering, classify obstacle type.
#
# CAMERA RECOGNITION  (Webots built-in neural network)
#   - camera.recognitionEnable() — no training required.
#   - Vehicle categories (case-insensitive): bus, car, truck, van, toyota,
#     lincoln, citroen, bmw, mercedes, suv. Own car (bmw) auto-skipped.
#   - Also recognises: traffic_light, sign (caution/order/speed/yield/highway),
#     obstacle (traffic cone, barrel, crash barrier). Buildings/trees ignored.
#   - Color-coded bounding boxes: red=vehicle, yellow=traffic_light,
#     cyan=sign, blue=obstacle.
#   - Classification runs every frame (manual + autonomous) for dataset labeling.
#   - Only camera objects within LIDAR_STOP_DISTANCE (15 m) are used for
#     emergency classification — prevents distant buses from triggering stops.
#
# OBSTACLE CLASSIFICATION & EVASION PHASE  (all modes)
#   - obstacle_type: VEHICLE | OBSTACLE | LIDAR_ONLY | OBJECT_AHEAD | ""
#   - vehicle_nearby: True when a vehicle is confirmed < 15 m by both LiDAR+camera.
#   - Evasion triggers ONLY for vehicles (hazard on + full brake).
#     Non-vehicle obstacles (cones, barriers) → full brake, no hazard, no evasion.
#   - evasion_phase state machine:
#       0 NORMAL   — no vehicle in stop range
#       1 EVADING  — vehicle_nearby is True
#       2 RETURNING — vehicle just cleared; lasts RETURNING_DURATION (5 s)
#
# SENSORS  (GPS + Gyro + Distance Sensors)
#   - GPS, Gyro, DS_ML, DS_MR, DS_FL, DS_FR, DS_RL, DS_RR enabled at startup;
#     each skipped gracefully if device not found in Webots world.
#   - Values read every frame and forwarded to the ADAS Monitor.
#
# ADAS MONITOR  (draw_adas_monitor)
#   - Separate OpenCV window "ADAS Monitor" toggled with key A.
#   - Displays: Speed, Angle, Brake (VEHICLE STATE section).
#   - Displays: GPS X/Y/Z in metres (GPS section).
#   - Displays: Gyro ωX/ωY/ωZ in rad/s (GYRO section).
#   - Displays: DS_ML, DS_MR, DS_FL, DS_FR, DS_RL, DS_RR with color coding
#     (green > 1000, orange 500–1000, red < 500). LIDAR row reserved.
#
# DATASET CAPTURE  (dataset_mode_v3 — Behavioral Cloning)
#   - Toggle with keyboard D or PS4 Triangle.
#   - Captures at 2 Hz after all control decisions (labels match current frame).
#   - Works in manual mode — captures expert driving demonstrations.
#   - CSV columns: session_id, timestamp, image_filename, steering_angle,
#     speed_kmh, brake, autonomous_mode, line_detected, gps_x, gps_y,
#     lidar_obstacle_detected, lidar_obstacle_distance, evasion_phase, obstacle_type.
#   - Images → data/behavioral_dataset/images/frame_XXXXXX.jpg
#   - Metadata → data/behavioral_dataset/measurements.csv (append mode)
#   - Frame counter resumes from existing files — toggling D never overwrites.
#
# INPUT / CONTROL
#   - Keyboard: arrows (speed/angle), S (autonomous), A (ADAS monitor),
#     C (capture image), D (dataset mode), P (debug panel), Q (quit).
#   - PS4 controller: left stick (steering), R2 (throttle), L2 (brake/reverse),
#     X (toggle autonomous), Square (save image), Triangle (dataset mode).
#   - Debounce: 0.1 s keyboard, 0.5 s PS4 buttons.
#
# DEBUG DISPLAYS
#   - "Self Driving Debug": annotated camera view with line, lane center, HUD,
#     obstacle status, evasion phase, and dataset capture indicator.
#   - "Yellow Mask Debug": binary ROI mask used for line detection.
#   - "PID Response Chart": live error (px), steering (rad), and speed (km/h) traces.
#   - "ADAS Monitor": sensor dashboard (speed, angle, brake, GPS, gyro, DS array).
#   - Speed overlay color: cyan = autonomous, green = manual.
#   - DEBUG_PANEL (key P) gates the Self Driving Debug and Yellow Mask windows.
#   - ADAS_MONITOR_ENABLED (key A) gates the ADAS Monitor window independently.
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
import dataset_mode_v3 as dataset_mode


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
BTN_X = 0
BTN_TRIANGLE = 2
BTN_SQUARE = 3

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
LIDAR_STOP_DISTANCE = 15.0  # meters


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

def draw_adas_monitor(speed, angle, brake, gps_vals, gyro_vals, ds_ml_val, ds_mr_val, ds_fl_val, ds_fr_val, ds_rl_val, ds_rr_val):
    W, H = 400, 650
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
            return (0, 0, 255)    # red — close
        if val < 1000:
            return (0, 165, 255)  # orange — approaching
        return (0, 255, 0)        # green — clear

    # Title
    cv2.putText(panel, "ADAS  MONITOR", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(panel, (8, 32), (W - 8, 32), (80, 80, 80), 1)

    # ── VEHICLE STATE ─────────────────────────────────
    section_header(54, "VEHICLE STATE")
    spd_color = (0, 255, 0) if speed < 60 else (0, 165, 255)
    brk_color = (0, 0, 255) if brake > 0.1 else (210, 210, 210)
    data_row(76,  "Speed",  f"{speed:.1f}  km/h", spd_color)
    data_row(99,  "Angle",  f"{angle:.4f}  rad")
    data_row(122, "Brake",  f"{brake:.2f}", brk_color)

    # ── GPS ───────────────────────────────────────────
    section_header(150, "GPS")
    if gps_vals is not None:
        data_row(172, "X", f"{gps_vals[0]:.3f}  m")
        data_row(195, "Y", f"{gps_vals[1]:.3f}  m")
        data_row(218, "Z", f"{gps_vals[2]:.3f}  m")
    else:
        data_row(182, "Status", "NOT AVAILABLE", (0, 80, 255))

    # ── GYRO ──────────────────────────────────────────
    section_header(246, "GYRO")
    if gyro_vals is not None:
        data_row(268, "wX", f"{gyro_vals[0]:.4f}  rad/s")
        data_row(291, "wY", f"{gyro_vals[1]:.4f}  rad/s")
        data_row(314, "wZ", f"{gyro_vals[2]:.4f}  rad/s")
    else:
        data_row(278, "Status", "NOT AVAILABLE", (0, 80, 255))

    # ── DISTANCE SENSORS ──────────────────────────────
    section_header(342, "DISTANCE SENSORS")
    ml_str = f"{ds_ml_val:.1f}" if ds_ml_val is not None else "--"
    mr_str = f"{ds_mr_val:.1f}" if ds_mr_val is not None else "--"
    fl_str = f"{ds_fl_val:.1f}" if ds_fl_val is not None else "--"
    fr_str = f"{ds_fr_val:.1f}" if ds_fr_val is not None else "--"
    rl_str = f"{ds_rl_val:.1f}" if ds_rl_val is not None else "--"
    rr_str = f"{ds_rr_val:.1f}" if ds_rr_val is not None else "--"
    data_row(364, "DS_ML  (mid-left)",  ml_str, ds_color(ds_ml_val))
    data_row(387, "DS_MR  (mid-right)", mr_str, ds_color(ds_mr_val))
    data_row(410, "DS_FL  (front-left)",  fl_str, ds_color(ds_fl_val))
    data_row(433, "DS_FR  (front-right)", fr_str, ds_color(ds_fr_val))
    data_row(456, "DS_RL  (rear-left)",   rl_str, ds_color(ds_rl_val))
    data_row(479, "DS_RR  (rear-right)",  rr_str, ds_color(ds_rr_val))

    # ── LIDAR (reserved) ──────────────────────────────
    section_header(507, "LIDAR")
    reserved_row(529, "[ reserved — front distance, angle ]")

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

    cv2.line(
        debug_frame,
        (desired_x, int(height * 0.45)),
        (desired_x, height),
        (255, 0, 255),
        3
    )

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

    print("Controller ready.")
    print("Keyboard S = toggle autonomous mode")
    print("Keyboard A = toggle ADAS monitoring panel")
    print("Keyboard D = toggle dataset mode")
    print("Keyboard P = toggle debug panel")
    print("Keyboard Q = quit")
    print("PS4 X = toggle autonomous mode")
    print("PS4 Triangle = toggle dataset mode")
    print("PS4 Square = save image")

    hazard_lights_on = False
    obstacle_status = "CLEAR"
    obstacle_type = ""        # camera-confirmed type of the blocking object
    vehicle_nearby = False    # True when a vehicle is confirmed < LIDAR_STOP_DISTANCE
    evasion_phase = 0         # 0=NORMAL, 1=EVADING, 2=RETURNING
    _evasion_clear_time = None
    RETURNING_DURATION = 5.0  # seconds to stay in RETURNING phase after vehicle clears

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
        # ALWAYS RUN YELLOW LINE DETECTION
        # This allows debugging in MANUAL and AUTO
        # ==============================

        line_detected, line_x, desired_x, debug_frame, yellow_roi = detect_yellow_line(frame)

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

        if line_detected:

            # ==========================================
            # VIRTUAL LANE CENTER
            # ==========================================
            # Yellow line is the LEFT boundary of the lane.
            # We estimate the center of the driving lane by
            # adding a horizontal offset to the right.
            # ==========================================


            # Estimated center of the right lane
            lane_center_x = line_x + LANE_OFFSET_PX

            # Keep inside image bounds
            lane_center_x = max(0, min(frame.shape[1] - 1, lane_center_x))

            # Real center of the camera/car
            camera_center_x = frame.shape[1] // 2

            # PID error
            error = lane_center_x - camera_center_x

            # Draw virtual lane center
            cv2.line(
                debug_frame,
                (lane_center_x, int(frame.shape[0] * 0.45)),
                (lane_center_x, frame.shape[0]),
                (255, 0, 255),
                3
            )

            cv2.putText(
                debug_frame,
                f"Line X: {line_x} | Lane Center: {lane_center_x} | Error: {error}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            
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
        if vehicle_nearby:
            evasion_phase = 1
            _evasion_clear_time = None
        elif evasion_phase == 1:
            # Vehicle just cleared — start return timer
            evasion_phase = 2
            _evasion_clear_time = current_time
        elif evasion_phase == 2 and _evasion_clear_time is not None:
            if (current_time - _evasion_clear_time) > RETURNING_DURATION:
                evasion_phase = 0
                _evasion_clear_time = None
        else:
            evasion_phase = 0

        # ==============================
        # AUTONOMOUS MODE — PID + LIDAR SAFETY OVERRIDE
        # ==============================

        if autonomous_mode:

            # ==========================================
            # LIDAR SAFETY OVERRIDE
            # ==========================================
            # LiDAR has priority over lane following.
            # If an obstacle is detected inside the stop distance,
            # the vehicle stops even if the yellow line is detected correctly.
            # ==========================================
            # ==========================================
            # EMERGENCY SAFETY LOGIC
            # ==========================================

            if lidar_obstacle and lidar_obstacle_distance < LIDAR_MAX_DISTANCE:

                brake = 0.0

                if lidar_obstacle_distance < LIDAR_STOP_DISTANCE:
                    # obstacle_type and vehicle_nearby already set by classification block.
                    # Evasion (hazard + stop) only for vehicles — cones/barrels just stop.
                    speed = 0
                    angle = 0.0
                    previous_angle = 0.0
                    brake = 1.0
                    hazard_lights_on = vehicle_nearby

                else:
                    # Object detected between 15 m and 30 m:
                    # slow down, but keep lane-following steering active
                    hazard_lights_on = False
                    obstacle_status = "OBJECT AHEAD"

                    if line_detected:
                        line_lost_time = None
                        pid_output = pid.compute(error, dt)

                        angle = 0.6 * previous_angle + 0.4 * pid_output
                        previous_angle = angle
                    else:
                        # Line lost while object is ahead.
                        # Fade steering instead of keeping old steering alive.
                        angle = previous_angle * 0.5

                    if abs(angle) < 0.03:
                        angle = 0.0

                    previous_angle = angle

                    speed = 0.5 * AUTONOMOUS_SPEED
                    brake = 0.0
                    

            else:
                # No emergency obstacle — resume normal autonomous lane following.
                hazard_lights_on = False
                brake = 0

                if line_detected:
                    # Yellow line visible — clear fallback timer and follow the lane.
                    line_lost_time = None

                    pid_output = pid.compute(error, dt)

                    # Smooth steering to reduce abrupt changes between frames.
                    angle = 0.6 * previous_angle + 0.4 * pid_output
                    previous_angle = angle

                    # Reduce speed proportionally when the lateral error is large.
                    speed = AUTONOMOUS_SPEED - min(abs(error) * 0.15, 10)
                    speed = max(speed, 8)

                    cv2.putText(
                        debug_frame,
                        f"AUTO | PID angle: {angle:.3f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                else:
                    # ==========================================
                    # LINE LOST LOGIC
                    # ==========================================

                    # Keep previous steering for a few frames
                    # but reduce it progressively.
                    angle = previous_angle * 0.5

                    # Slow down while searching
                    speed = 10

                    # Small steering values become zero
                    # to avoid oscillation.
                    if abs(angle) < 0.03:
                        angle = 0.0

                    # Record first lost frame
                    if line_lost_time is None:
                        line_lost_time = current_time

                    
                    # Countdown logic
                    time_remaining = max(0.0, 10.0 - (current_time - line_lost_time))
                    countdown = int(np.ceil(time_remaining))

                    cv2.putText(
                        debug_frame,
                        "AUTO | Searching yellow line",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    countdown_color = (0, 100, 255) if countdown > 2 else (0, 0, 255)

                    cv2.putText(
                        debug_frame,
                        f"Going back to Manual MODE: {countdown}",
                        (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        countdown_color,
                        2
                    )

                    # If line remains lost too long
                    if time_remaining <= 0:
                        autonomous_mode = False
                        line_lost_time = None
                        pid.reset()
                        previous_angle = 0.0
                        angle = 0.0

                        print("Line lost — automatically reverted to MANUAL mode")

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

        phase_labels = {0: "NORMAL", 1: "EVADING", 2: "RETURNING"}
        phase_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)}
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
            f"Evasion Phase: {evasion_phase} ({phase_labels[evasion_phase]})",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            phase_colors[evasion_phase],
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
            line_detected=line_detected,
            gps_vals=gps_vals,
            lidar_obstacle_detected=lidar_obstacle,
            lidar_obstacle_distance=lidar_obstacle_distance,
            evasion_phase=evasion_phase,
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
            draw_adas_monitor(speed, angle, brake, gps_vals, gyro_vals, ds_ml_val, ds_mr_val, ds_fl_val, ds_fr_val, ds_rl_val, ds_rr_val)

        # Display image in Webots display
        if DEBUG_PANEL:
            display_image(display_img, display_frame)

            # Optional OpenCV debug window
            cv2.imshow("Self Driving Debug", display_frame)
            cv2.imshow("Yellow Mask Debug", cv2.resize(yellow_roi, (640, 300)))

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