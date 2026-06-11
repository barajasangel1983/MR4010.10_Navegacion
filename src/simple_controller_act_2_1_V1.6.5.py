# ============================================================
# V1.6
# Webots Self-Driving Controller
# Grayscale Brightness Threshold + Hough Transform + PID + PS4 Bluetooth
# Variant of V1.2SDF_ — replaces the HSV yellow mask with a grayscale
# brightness threshold. Simpler pipeline; trades color selectivity for
# robustness against HSV hue shifts caused by Webots lighting changes.

# Sign finder with 3 ROIs with CNN based in webots
# ============================================================

from controller import Display, Keyboard, Lidar
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time
import pygame
import pickle
import json
import tensorflow as tf
from skimage.feature import hog
from collections import deque


# ==============================
# MOVING AVERAGE LINEX FROM PREVIOUS FRAMES
# ==============================
line_x_history = deque(maxlen=10)


# ==============================
# SVM MODELS
# ==============================

VEHICLE_MODEL_PATH = r"D:\ML\Projects\Project_5_MR4010.10_Navegacion\MR4010.10_Navegacion\models\vehicle_svm_hog.pkl"
PEDESTRIAN_MODEL_PATH = r"D:\ML\Projects\Project_5_MR4010.10_Navegacion\MR4010.10_Navegacion\models\pedestrian_svm_hog.pkl"

# ==============================
# CNN SIGN CLASSIFIER MODEL
# ==============================
# This model was trained with Webots images organized by folders:
# no_sign, nopedestrian, speed, stop, yield.
# Runtime uses a hybrid pipeline: color/contour heuristic finds a candidate
# crop first, then the CNN classifies that crop. This avoids classifying a large
# ROI that is mostly background.
ENABLE_SIGN_DETECTION = True

SIGN_MODEL_PATH = r"D:\ML\Projects\Project_5_MR4010.10_Navegacion\MR4010.10_Navegacion\models\webots_sign_cnn.keras"
SIGN_CLASSES_PATH = r"D:\ML\Projects\Project_5_MR4010.10_Navegacion\MR4010.10_Navegacion\models\webots_sign_classes.json"

SIGN_IMG_SIZE = (64, 64)
SIGN_CONFIDENCE_THRESHOLD = 0.75
SIGN_DETECTION_INTERVAL = 0.50


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

SIGN_DEBUG_PERSIST_SEC = 3.0

# Three independent horizontal ROI bands: (name, x1_ratio, x2_ratio)
SIGN_ROIS = [
    ("LEFT",   0.00, 1/3),
    ("CENTER", 1/3,  2/3),
    ("RIGHT",  2/3,  1.00),
]

SIGN_ROI_COLORS = {
    "LEFT":   (0, 128, 255),  # orange
    "CENTER": (0, 255, 255),  # cyan
    "RIGHT":  (0, 200, 0),    # green
}

# Per-ROI heuristic debug state
_sign_debug = {name: {"boxes": [], "time": 0.0} for name, _, _ in SIGN_ROIS}

# Per-ROI CNN hold state (last confirmed non-no_sign result)
_cnn_held = {
    name: {"label": "no_sign", "confidence": 0.0, "box": None, "roi_box": None, "time": 0.0}
    for name, _, _ in SIGN_ROIS
}

# PS4 button map
BTN_X = 0
BTN_CIRCLE = 1
BTN_TRIANGLE = 2
BTN_SQUARE = 3
BTN_L1 = 4
BTN_R1 = 5
BTN_SHARE = 8
BTN_OPTIONS = 9
BTN_PS = 10

#Virtual Lane
LANE_OFFSET_PX = 110

#Blue dot location/ reference to screen
Y_REF_RATIO = 0.85

# CANNY + HOUGH / / Jump rejection to avoid edge-to-edge oscillation
MAX_LINE_JUMP_PX = 100

#Line MAx SLOPE allowed for detection
MAX_SLOPE=0.15

#CAMERA SETTINGS
PEDESTRIAN_IMG_SIZE = (18, 36)  # OpenCV resize usa (width, height)
CAMERA_WIDTH = 620
CAMERA_HEIGHT = 320


# OBJECT DETECTION CONFIG
ENABLE_VEHICLE_DETECTION = False
ENABLE_PEDESTRIAN_DETECTION = False

# ==============================
# LIDAR OBSTACLE DETECTION CONFIG
# ==============================

# Enables or disables LiDAR-based obstacle detection.
ENABLE_LIDAR_OBSTACLE_DETECTION = True

# Maximum detection distance required by the assignment.
# Any obstacle farther than this value will be ignored.
LIDAR_MAX_DISTANCE = 25.0  # meters

# Front-center LiDAR detection window.
# Requirement says 20 or 30 degrees. Use 30 for a wider and safer zone.
LIDAR_CENTER_FOV_DEG = 30

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


def make_line(height, slope, intercept):
    # Project a y=mx+b line onto two fixed row heights to get drawable endpoints.
    y1 = int(height * 0.95)
    y2 = int(height * 0.75)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [x1, y1, x2, y2]

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

# Loading SVM Models
def load_svm_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
        
#Vehicles detection
def detect_objects_svm(frame,
                       model,
                       roi_y_start=0.35,
                       window_size=(64, 64),
                       step=32):

    height, width, _ = frame.shape

    detections = []

    win_w = window_size[0]
    win_h = window_size[1]

    y_start = int(height * roi_y_start)
    #defining mask for inspection
    x_start = int(width * 0.15)
    x_end = int(width * 0.85)

    for y in range(y_start, height - win_h, step):

        for x in range(x_start, x_end - win_w, step):

            crop = frame[y:y + win_h, x:x + win_w]

            crop = cv2.resize(crop, window_size)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            features = hog(
                gray,
                orientations=11,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                transform_sqrt=False,
                visualize=False,
                feature_vector=True
            )

            features = features.reshape(1, -1)

            pred = model.predict(features)[0]

            if pred == 1:
                detections.append((x, y, win_w, win_h))

    return detections



# HOG feature extraction for pedestrians — matches training-time parameters exactly
def extract_pedestrian_hog_features(img):
    img = cv2.resize(img, PEDESTRIAN_IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        visualize=False,
        feature_vector=True
    )

    return features.reshape(1, -1)

# NMS — Non-Maximum Suppression: eliminates overlapping detection boxes by score
def non_max_suppression(boxes, scores, overlap_thresh=0.25):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes).astype(float)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(
            idxs,
            np.concatenate((
                [len(idxs) - 1],
                np.where(overlap > overlap_thresh)[0]
            ))
        )

    return boxes[pick].astype(int).tolist()

# HOG sliding-window detection for pedestrians
def detect_pedestrians_svm(frame, pedestrian_model):
    height, width, _ = frame.shape

    boxes = []
    scores = []

    model_w = 18
    model_h = 36

    #pedestrian_sizes = [
    #    (30, 60),
    #    (33, 66),
    #    (36, 72)
    #]
    pedestrian_sizes = [
        (30, 65),
        (45, 80),
        (60, 95)
    ]

    confidence_threshold = 0.4
    yellow_ratio_limit = 0.10
    edge_density_min = 0.08
    step = 48

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    y_start = int(height * 0.3)
    y_end = int(height * 0.9)

    x_start = int(width * 0.25)
    x_end = int(width * 0.75)

    for win_w, win_h in pedestrian_sizes:

        for y in range(y_start, y_end - win_h, step):
            for x in range(x_start, x_end - win_w, step):

                candidate = gray[y:y + win_h, x:x + win_w]
                candidate_bgr = frame[y:y + win_h, x:x + win_w]

                # ==============================
                # YELLOW ROAD MARKING FILTER
                # ==============================

                hsv = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2HSV)

                yellow_mask = cv2.inRange(
                    hsv,
                    np.array([20, 80, 80]),
                    np.array([40, 255, 255])
                )

                yellow_ratio = np.sum(yellow_mask > 0) / (win_w * win_h)

                if yellow_ratio > yellow_ratio_limit:
                    continue

                # ==============================
                # VERTICAL POSITION FILTER
                # ==============================

                box_center_y = y + win_h / 2

                if box_center_y < height * 0.35:
                    continue

                if box_center_y > height * 0.88:
                    continue

                # ==============================
                # RESIZE TO TRAINING SIZE
                # ==============================

                candidate_resized = cv2.resize(
                    candidate,
                    (model_w, model_h)
                )

                # ==============================
                # EDGE DENSITY FILTER
                # ==============================

                edges = cv2.Canny(candidate_resized, 50, 150)

                edge_density = np.sum(edges > 0) / (model_w * model_h)

                if edge_density < edge_density_min:
                    continue

                # ==============================
                # HOG FEATURES
                # ==============================

                features = hog(
                    candidate_resized,
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    transform_sqrt=True,
                    visualize=False,
                    feature_vector=True
                ).reshape(1, -1)

                score = pedestrian_model.decision_function(features)[0]

                if score > confidence_threshold:
                    boxes.append((x, y, win_w, win_h))
                    scores.append(score)

    final_boxes = non_max_suppression(
        boxes,
        scores,
        overlap_thresh=0.25
    )

    return final_boxes


# ==============================
# CNN SIGN CLASSIFICATION — HYBRID PIPELINE
# ==============================
# Pipeline used at runtime:
#   1. Search only the right-side ROI where signs appear.
#   2. Canny edge detection + contour shape analysis finds candidates.
#      Shape-based detection is lighting/color independent — works in Webots
#      where the low-saturation scene makes HSV color masks unreliable.
#   3. Each candidate crop is sent to the CNN classifier.
#   4. Best non-no_sign result is kept.
# ==============================

def load_sign_model(model_path, classes_path):
    """
    Load the trained Keras CNN sign classifier and the class labels.

    The model was trained using image_dataset_from_directory(), so the class
    names come from the dataset folder names.
    """
    model = tf.keras.models.load_model(model_path)

    with open(classes_path, "r") as f:
        class_names = json.load(f)

    print("Sign CNN model loaded.")
    print("Sign classes:", class_names)

    return model, class_names


def clamp(value, min_value, max_value):
    """Clamp value to the inclusive range [min_value, max_value]."""
    return max(min_value, min(max_value, value))


def detect_sign_candidates(img_bgr, max_candidates=4):
    """
    Find candidate traffic-sign boxes using Canny edges + contour shape analysis.

    Replaces the HSV color heuristic which was unreliable in Webots: the scene
    has globally low saturation, making red/white masks match the background.

    Shape filters used (all traffic signs share these properties):
      - Area:        150 px² – 50% of ROI  (small distant signs to nearby large ones)
      - Vertices:    3–15 after polygon approximation (triangle, quad, octagon, circle)
      - Solidity:    > 0.75  (signs are convex or nearly convex)
      - Aspect ratio: 0.25 – 3.5  (excludes poles and wide road markings)
      - Fill ratio:  > 0.35  (contour area / bounding rect — eliminates thin edge groups)

    Returns:
        boxes: list of (x, y, w, h) in local ROI coordinates, sorted by area desc
        edges: Canny edge image for the debug window
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE: amplifies local contrast so sign faces stand out from background
    # even when absolute brightness difference is small (Webots low-saturation issue).
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light Gaussian blur — less aggressive than bilateral, preserves sign outlines.
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Low thresholds: Webots sign edges are weak, so we need to catch them early.
    edges   = cv2.Canny(blurred, 30, 90)

    # Dilate to close small gaps in the sign outline from rendering artifacts.
    kernel        = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = gray.shape
    image_area   = w_img * h_img

    valid_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 150 or area > image_area * 0.50:
            continue

        peri = cv2.arcLength(contour, True)
        if peri == 0:
            continue

        # Polygon approximation — 4% of perimeter as epsilon.
        approx   = cv2.approxPolyDP(contour, 0.04 * peri, True)
        vertices = len(approx)

        # Triangle (yield=3), quad (speed limit=4), hexagon/octagon (stop=6-8),
        # circle approximation (no-pedestrian ≈ 8-15).
        if vertices < 3 or vertices > 15:
            continue

        # Solidity: contour area / convex hull area. Signs are mostly convex.
        hull      = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1 or (area / hull_area) < 0.45:
            continue

        x, y, w, h  = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)

        if aspect_ratio < 0.25 or aspect_ratio > 3.5:
            continue

        # Fill ratio: very low = thin edge group (pole, wire), not a sign body.
        if area / (w * h) < 0.15:
            continue

        valid_boxes.append((x, y, w, h, area))

      
    # ==============================
    # FALLBACK: white rectangular signs
    # Good for SPEED LIMIT signs
    # ==============================
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 120]),
        np.array([180, 80, 255])
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    white_contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in white_contours:
        area = cv2.contourArea(c)

        if area < 300 or area > image_area * 0.30:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / max(h, 1)

        # speed sign is usually vertical rectangle
        if 0.45 < aspect_ratio < 1.20 and h > 25 and w > 20:
            valid_boxes.append((x, y, w, h, area))
    

    # Deduplicate boxes with nearly the same centroid — Canny finds both the
    # outer outline and inner features (text, border ring) of the same sign.
    valid_boxes.sort(key=lambda b: b[4], reverse=True)
    deduped = []
    for box in valid_boxes:
        x, y, w, h, a = box
        cx, cy = x + w // 2, y + h // 2
        duplicate = any(
            abs(cx - (dx + dw // 2)) < 20 and abs(cy - (dy + dh // 2)) < 20
            for dx, dy, dw, dh, _ in deduped
        )
        if not duplicate:
            deduped.append(box)

    boxes = [(x, y, w, h) for x, y, w, h, _ in deduped[:max_candidates]]

    return boxes, edges

def crop_with_margin(img_bgr, box, margin_ratio=0.60):
    """
    Crop the candidate sign box with margin and resize to CNN input size.

    margin_ratio=0.60 gives speed-limit signs enough context so the
    full SPEED / LIMIT / number area is included in the crop.

    Returns:
        crop_resized: image ready for CNN input, BGR format
        crop_box: (x, y, w, h) of the expanded crop in local coordinates
    """
    h_img, w_img, _ = img_bgr.shape
    x, y, w, h = box

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = clamp(x - margin_x, 0, w_img - 1)
    y1 = clamp(y - margin_y, 0, h_img - 1)
    x2 = clamp(x + w + margin_x, 0, w_img)
    y2 = clamp(y + h + margin_y, 0, h_img)

    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    crop_resized = cv2.resize(crop, SIGN_IMG_SIZE)

    return crop_resized, (x1, y1, x2 - x1, y2 - y1)


def classify_sign_frame(frame_bgr, sign_model, sign_class_names,
                         x1_ratio=2/3, x2_ratio=1.0, region_name="RIGHT"):
    """
    Hybrid pipeline for one horizontal ROI band.

    Returns: label, confidence, sign_box (full-frame), search_roi_box, roi_img, candidate_mask
    """
    global _sign_debug

    height, width, _ = frame_bgr.shape

    roi_x1 = int(width * x1_ratio)
    roi_y1 = 0
    roi_x2 = int(width * x2_ratio)
    roi_y2 = height

    search_roi_box = (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1)
    roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

    candidates, candidate_mask = detect_sign_candidates(roi)

    # Update per-region debug persistence state.
    now = time.time()
    region_debug = _sign_debug[region_name]
    if candidates:
        region_debug["boxes"] = list(candidates)
        region_debug["time"]  = now

    # Grid fallback when heuristic finds nothing.
    if not candidates:
        rh, rw, _ = roi.shape
        grid_boxes = []
        for scale in [0.6, 0.85]:
            cw_g = int(rw * scale)
            ch_g = int(rh * scale)
            for gy in range(0, rh - ch_g + 1, max(1, ch_g // 2)):
                for gx in range(0, rw - cw_g + 1, max(1, cw_g // 2)):
                    grid_boxes.append((gx, gy, cw_g, ch_g))
        candidates = grid_boxes

    best_label      = "no_sign"
    best_confidence = 0.0
    best_sign_box   = None

    for candidate_box in candidates:
        crop_resized, crop_box_roi = crop_with_margin(roi, candidate_box)

        if crop_resized is None:
            continue

        # Model was trained on RGB images (TensorFlow image_dataset_from_directory).
        crop_rgb     = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(crop_rgb.astype("float32"), axis=0)

        predictions = sign_model.predict(input_tensor, verbose=0)[0]
        class_id    = int(np.argmax(predictions))
        confidence  = float(predictions[class_id])
        label       = sign_class_names[class_id]

        if label != "no_sign" and confidence > best_confidence:
            best_label      = label
            best_confidence = confidence
            cx, cy, cw, ch  = crop_box_roi
            best_sign_box   = (roi_x1 + cx, roi_y1 + cy, cw, ch)

    return best_label, best_confidence, best_sign_box, search_roi_box, roi, candidate_mask


def draw_sign_detection(debug_frame, label, confidence, sign_box, search_roi_box=None,
                         region_name="RIGHT", roi_color=(0, 200, 0), y_offset=250):
    """
    Draw the sign detection result for one ROI region.

    Colored box = search ROI boundary.
    Tighter box = CNN candidate crop.
    Text        = SIGN <REGION>: label (confidence).
    """
    if search_roi_box is not None:
        rx, ry, rw, rh = search_roi_box
        cv2.rectangle(debug_frame, (rx, ry), (rx + rw, ry + rh), roi_color, 2)

    text_color = roi_color if label == "no_sign" else roi_color

    if sign_box is None:
        cv2.putText(
            debug_frame,
            f"SIGN {region_name}: {label} ({confidence:.2f})",
            (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2
        )
        return

    x, y, w, h = sign_box
    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), roi_color, 2)

    cv2.putText(
        debug_frame,
        f"SIGN {region_name}: {label} ({confidence:.2f})",
        (30, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        roi_color,
        2
    )

    cv2.putText(
        debug_frame,
        f"{label} {confidence:.2f}",
        (x, max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        roi_color,
        2
    )

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

    x_intersections = []
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

        if abs(avg_slope) < 1e-6:
            yellow_line_prev = None
            line_x_history.clear()
            return False, None, desired_x, debug_frame, yellow_roi

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
    global ENABLE_SIGN_DETECTION
    global _sign_debug, _cnn_held

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

    # Loading models SVM Models for vision
    vehicle_model = load_svm_model(VEHICLE_MODEL_PATH)
    pedestrian_model = load_svm_model(PEDESTRIAN_MODEL_PATH)

    # Loading CNN model for Webots traffic sign classification.
    sign_model = None
    sign_class_names = []

    if ENABLE_SIGN_DETECTION:
        sign_model, sign_class_names = load_sign_model(
            SIGN_MODEL_PATH,
            SIGN_CLASSES_PATH
        )


    # This is for the SVM pedestrian time control
    previous_time = time.time()
    frame_count = 0
    last_detection_time = 0

    vehicle_boxes = []
    pedestrian_boxes = []

    # Last known sign classification result per ROI region.
    # Updated periodically (SIGN_DETECTION_INTERVAL) to limit CNN inference cost.
    sign_results = {
        name: {"label": "no_sign", "confidence": 0.0, "box": None, "roi_box": None}
        for name, _, _ in SIGN_ROIS
    }
    last_sign_detection_time = 0


    print("Controller ready.")
    print("Keyboard S = toggle autonomous mode")
    print("PS4 X = toggle autonomous mode")
    print("Square = save image")
    print("Keyboard G = toggle CNN sign detection")
    print("PS4 Triangle = toggle CNN sign detection")

    #Hazard lights control
    #
    hazard_lights_on = False
    obstacle_status = "CLEAR"

    while robot.step() != -1:
        current_time = time.time()
        dt = current_time - previous_time
        previous_time = current_time

        frame = get_image(camera)
        frame_count += 1

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

                elif key == ord("A"):
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

                elif key == ord("G"):
                    ENABLE_SIGN_DETECTION = not ENABLE_SIGN_DETECTION
                    print(f"Sign detection: {ENABLE_SIGN_DETECTION}")

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

            if joystick.get_button(BTN_SQUARE):
                if "ps_square" not in last_press or current_time - last_press["ps_square"] > 0.5:
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    file_name = current_datetime + ".png"
                    camera.saveImage(os.path.join(os.getcwd(), file_name), 1)
                    print("Image taken from PS controller")
                    last_press["ps_square"] = current_time

            if joystick.get_button(BTN_TRIANGLE):
                if "ps_triangle" not in last_press or current_time - last_press["ps_triangle"] > 0.5:
                    ENABLE_SIGN_DETECTION = not ENABLE_SIGN_DETECTION
                    last_press["ps_triangle"] = current_time
                    print(f"Sign detection: {ENABLE_SIGN_DETECTION}")

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
        # RUN HEAVY SVM DETECTION ONLY EVERY 1 second
        # ==========================================

        if current_time - last_detection_time > 1.0:

            last_detection_time = current_time
            
            if ENABLE_VEHICLE_DETECTION:
                
                vehicle_boxes = detect_objects_svm(
                    frame,
                    vehicle_model,
                    roi_y_start=0.35,
                    window_size=(64, 64),
                    step=64
                )

            if ENABLE_PEDESTRIAN_DETECTION:

                pedestrian_boxes = detect_pedestrians_svm(
                    frame,
                    pedestrian_model
                )

        # ==========================================
        # RUN CNN SIGN CLASSIFICATION PERIODICALLY
        # ==========================================
        # This runs independently from the heavy SVM block.
        # The runtime pipeline first finds a sign candidate inside a large ROI,
        # then sends only the tight candidate crop to the CNN classifier.

        if (
            ENABLE_SIGN_DETECTION
            and sign_model is not None
            and current_time - last_sign_detection_time > SIGN_DETECTION_INTERVAL
        ):
            last_sign_detection_time = current_time

            debug_rows = []
            for roi_name, x1_r, x2_r in SIGN_ROIS:
                lbl, conf, sbox, roi_box, roi_img, canny_mask = classify_sign_frame(
                    frame, sign_model, sign_class_names, x1_r, x2_r, roi_name
                )
                sign_results[roi_name] = {
                    "label": lbl, "confidence": conf,
                    "box": sbox, "roi_box": roi_box
                }

                # Update 3-second hold for this region if a real sign was found.
                if lbl != "no_sign" and conf >= SIGN_CONFIDENCE_THRESHOLD:
                    _cnn_held[roi_name] = {
                        "label": lbl, "confidence": conf,
                        "box": sbox, "roi_box": roi_box, "time": current_time
                    }

                # Build this region's debug row: [ROI crop | Canny mask]
                color = SIGN_ROI_COLORS[roi_name]
                region_dbg = _sign_debug[roi_name]
                show_boxes = (
                    region_dbg["boxes"]
                    if (current_time - region_dbg["time"]) < SIGN_DEBUG_PERSIST_SEC
                    else []
                )
                pw, ph = 213, 200
                roi_panel   = cv2.resize(roi_img, (pw, ph))
                canny_panel = cv2.cvtColor(cv2.resize(canny_mask, (pw, ph)), cv2.COLOR_GRAY2BGR)
                sx = pw / roi_img.shape[1]
                sy = ph / roi_img.shape[0]
                for bx, by, bw, bh in show_boxes:
                    cv2.rectangle(roi_panel,
                        (int(bx * sx), int(by * sy)),
                        (int((bx + bw) * sx), int((by + bh) * sy)),
                        color, 2)
                cv2.putText(roi_panel, roi_name, (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                debug_rows.append(np.hstack([roi_panel, canny_panel]))

            cv2.imshow("Sign Heuristic Debug", np.vstack(debug_rows))

        for x, y, w, h in vehicle_boxes:
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(debug_frame, "Vehicle", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        

        for x, y, w, h in pedestrian_boxes:

            cv2.rectangle(
                debug_frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                debug_frame,
                "Pedestrian",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1
            )   

        # ==========================================
        # DRAW CNN SIGN CLASSIFIER RESULT
        # ==========================================
        # Draw search ROI, candidate bounding box, and classifier label.
        # If confidence is lower than the threshold, show the prediction as uncertain.

        if ENABLE_SIGN_DETECTION and sign_model is not None:
            for i, (roi_name, _, _) in enumerate(SIGN_ROIS):
                color  = SIGN_ROI_COLORS[roi_name]
                held   = _cnn_held[roi_name]
                result = sign_results[roi_name]

                # Use 3-second held result when a real sign was recently confirmed.
                if (current_time - held["time"]) < SIGN_DEBUG_PERSIST_SEC and held["label"] != "no_sign":
                    disp_label = held["label"]
                    disp_conf  = held["confidence"]
                    disp_box   = held["box"]
                    disp_roi   = held["roi_box"]
                elif result["box"] is None or result["confidence"] >= SIGN_CONFIDENCE_THRESHOLD:
                    disp_label = result["label"]
                    disp_conf  = result["confidence"]
                    disp_box   = result["box"]
                    disp_roi   = result["roi_box"]
                else:
                    disp_label = f"uncertain: {result['label']}"
                    disp_conf  = result["confidence"]
                    disp_box   = result["box"]
                    disp_roi   = result["roi_box"]

                draw_sign_detection(
                    debug_frame, disp_label, disp_conf, disp_box, disp_roi,
                    region_name=roi_name, roi_color=color, y_offset=250 + i * 22
                )
        else:
            cv2.putText(
                debug_frame,
                "SIGN CNN: OFF",
                (30, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (120, 120, 120),
                2
            )

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
                    pedestrian_detected = len(pedestrian_boxes) > 0

                    speed = 0
                    angle = 0.0
                    previous_angle = 0.0
                    brake = 1.0

                    if pedestrian_detected:
                        hazard_lights_on = False
                        obstacle_status = "PEDESTRIAN"
                    else:
                        hazard_lights_on = True
                        obstacle_status = "BARREL"

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
                obstacle_status = "CLEAR"
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

        cv2.putText(
            debug_frame,
            f"Obstacle Type: {obstacle_status}",
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
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
            (debug_frame.shape[1] - 210, debug_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            speed_color,
            2
        )

        display_frame = cv2.resize(debug_frame, (640, 300))

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

        # Display image in Webots display
        display_image(display_img, display_frame)

        # Optional OpenCV debug window
        cv2.imshow("Self Driving Debug", display_frame)
        cv2.imshow("Yellow Mask Debug", cv2.resize(yellow_roi, (640, 300)))

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