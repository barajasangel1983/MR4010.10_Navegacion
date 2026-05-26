# ============================================================
# Actividad 2.2 - Deteccion de Peatones con SVM + LiDAR
# Equipo 17 - Navegacion Autonoma
#
# Punto de partida: este archivo arranca como una copia EXACTA
# del controlador de la Actividad 2.1 (seguidor de linea con
# PID, deteccion de linea amarilla con HSV/grayscale, modos
# manual y autonomo con teclado y PS4).
#
# Sobre esa base se ira agregando, paso a paso:
#   - Lectura del LiDAR Sick LMS 291 frontal (FOV 20-30 grados,
#     rango maximo 20 metros).
#   - Carga del modelo SVM (pedestrian_svm.pkl) entrenado en
#     train_pedestrian_svm.py para clasificar peatones a partir
#     de descriptores HOG.
#   - Busqueda por ventanas deslizantes (Sliding Window Search)
#     sobre la imagen de la camara cuando el LiDAR detecte un
#     obstaculo cercano.
#   - Maquina de estados de obstaculos:
#         peaton  -> freno de emergencia
#         barril  -> freno de emergencia + luces intermitentes
#
# Cada modificacion respecto al 2.1 estara marcada con un
# comentario "# [2.2]" para que sea facil identificar en
# el reporte que se agrego o cambio.
# ============================================================

# ============================================================
# V1.4
# Webots Self-Driving Controller
# Grayscale Brightness Threshold + Hough Transform + PID + PS4 Bluetooth
# Variant of V1.2SDF_ - replaces the HSV yellow mask with a grayscale
# brightness threshold. Simpler pipeline; trades color selectivity for
# robustness against HSV hue shifts caused by Webots lighting changes.
# ============================================================

from controller import Display, Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time
import pygame
import joblib                          # [2.2 v6] cargar el modelo SVM entrenado
from skimage.feature import hog        # [2.2 v6] mismo HOG que en train_pedestrian_svm.py


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
AUTONOMOUS_SPEED = 60  # [2.2 v4] valor original 2.1 - camara ahora es 640x320, similar al original
STEERING_GAIN_LIMIT = 0.7 #before 0.5

# PID tuning
KP = 0.0015   # [2.2 v4] valor original 2.1 - camara 640x320 escala parecido al original
KI = 0.000
KD = 0.0003   # [2.2 v4] valor original 2.1

# Yellow line smoothing
yellow_line_prev = None
alpha_smooth = 0.25

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
LANE_OFFSET_PX = 105   # [2.2 v4] valor original 2.1 - camara 640x320 muy cercana al original

#Blue dot location/ reference to screen
Y_REF_RATIO = 0.75

# CANNY + HOUGH / / Jump rejection to avoid edge-to-edge oscillation
MAX_LINE_JUMP_PX = 100  # [2.2 v4] valor original 2.1 - camara 640x320 vuelve a la escala original

#Line MAx SLOPE allowed for detection
MAX_SLOPE=0.15

# ==============================
# [2.2 v5] LIDAR CONFIG
# El BmwX5 trae un SickLms291 montado en sensorsSlotFront.
# Por defecto escanea 180 grados con 180 puntos y rango maximo 80m.
# Aqui filtramos solo el FOV frontal y un rango util corto.
# ==============================
LIDAR_FRONT_FOV_DEG = 80      # [2.2 v7.11] subido de 70 a 80 (+-40deg) - mas sensible, cerca del original 90
LIDAR_MAX_DETECT_M  = 20.0    # ignoramos lo que este mas lejos que esto
LIDAR_MIN_VALID_M   = 0.30    # ignoramos lecturas demasiado cercanas (ruido / propio paragolpes)

# ==============================
# [2.2 v6] SVM + HOG CONFIG
# Mismos parametros que en train_pedestrian_svm.py.
# Cualquier diferencia rompe la inferencia (vector de features incompatible).
# ==============================
# [2.2 v6] Ruta absoluta basada en la ubicacion del script.
# Webots <extern> no garantiza cwd igual al folder del controlador.
SVM_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pedestrian_svm.pkl"
)
SVM_WIN_W, SVM_WIN_H = 64, 128
SVM_HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True,
)
SVM_SCALES = [1.0, 1.5]          # [2.2 v6.1] quitamos 0.75 para reducir falsos positivos y carga
SVM_STEP_X = 24                  # [2.2 v6.1] antes 16 - menos ventanas horizontales
SVM_STEP_Y = 32                  # [2.2 v6.1] antes 24 - menos ventanas verticales
SVM_ROI_HALF_WIDTH_PX = 100      # solo escanear esta franja a cada lado del centro LiDAR
SVM_RUN_EVERY_N_FRAMES = 6       # [2.2 v6.1] antes 3 - mitad de frecuencia para ganar fps
SVM_SCORE_THRESHOLD = 1.0        # [2.2 v6.1] descartar detecciones con margen < 1.0
SVM_NMS_IOU = 0.30               # [2.2 v6.1] umbral IoU para non-maximum suppression
# [2.2 v8.4] No muestrear el cielo/edificios. Empezar el sliding window
# Y a partir de este % del alto del frame. 0.25 => salta el 25% superior.
SVM_Y_START_MIN_RATIO = 0.25

# ==============================
# [2.2 v7] OBSTACLE STATE MACHINE CONFIG
# Distingue PEATON vs BARRIL usando la combinacion LiDAR + SVM:
#   - Si el SVM detecta peaton en STATE_SVM_HITS_TO_CONFIRM scans seguidos -> PEDESTRIAN_STOP
#   - Si el SVM corre y NO detecta peaton en STATE_SVM_MISSES_TO_CONFIRM scans seguidos
#     pero el LiDAR sigue viendo algo cerca -> BARREL_STOP
# Usa histeresis (brake_dist vs resume_dist) para no oscilar.
# ==============================
STATE_BRAKE_DIST_M           = 8.0   # entrar en STOP cuando el LiDAR ve algo a <= este distancia
STATE_RESUME_DIST_M          = 12.0  # volver a NORMAL cuando el LiDAR esta libre o > esta distancia
STATE_SVM_HITS_TO_CONFIRM    = 2     # cuantos scans positivos seguidos para confirmar PEATON
STATE_SVM_MISSES_TO_CONFIRM  = 3     # cuantos scans negativos seguidos con LiDAR cerca -> BARRIL

# [2.2 v7.1] Heuristica geometrica para descartar falsos positivos del SVM.
# El SVM tiende a confundir formas verticales (barril, postes) con peatones.
# Un peaton real visto desde la camara delantera del carro deberia ser ALTO y
# su parte superior (top de la bbox) deberia caer en la parte alta de la imagen
# (cerca del horizonte). Un barril/poste corto cae siempre en la parte baja.
STATE_PED_MIN_BBOX_HEIGHT_PX = 100   # altura minima del bbox para contar como peaton real
STATE_PED_MAX_TOP_Y_RATIO    = 0.55  # el top de la bbox debe estar arriba de este % del alto

# [2.2 v7.1] Intensidad de freno cuando estamos en estado STOP
STATE_BRAKE_INTENSITY = 1.0          # 0=nada, 1=freno maximo

# [2.2 v7.8] Tiempo minimo parado. Subido a 1.5s - balance entre frenado completo y agilidad.
STATE_MIN_STOP_DURATION_S = 1.5    # antes 1.0 (v7.7), 1.3 (v7.6), 1.0 (v7.5), 1.5 (v7.4), 3.0 (v7.3)

# [2.2 v7.8] Tiempo MAXIMO parado. Subido a 4.0s para dar mas tiempo al obstaculo a moverse antes del bypass.
STATE_MAX_STOP_DURATION_S = 4.0    # antes 3.5 (v7.7), 4.5 (v7.6), 3.0 (v7.5), 6.0 (v7.4)

# [2.2 v7.4] Maniobra de bordeo cuando expira el max timeout:
# Se agrega un sesgo angular al PID durante BYPASS_DURATION_S para alejarse del
# lado donde el LiDAR vio el obstaculo. Magnitud en radianes.
BYPASS_ANGLE_MAGNITUDE = 0.30      # ~17 grados, suficiente para esquivar
BYPASS_DURATION_S      = 2.0

# ==============================
# [2.2 v8] DETECCION POR COLOR (barril vs peaton)
# El HOG+SVM se confunde mucho entre barriles y peatones (ambos son verticales).
# Como en el mundo de Webots los barriles son naranja-oxido bien distintivos, usamos
# color en HSV para discriminar de forma mucho mas confiable.
#
# Cuando el LiDAR detecta obstaculo cercano, tomamos un parche pequeno de la camara
# en la columna donde apunta el LiDAR, y vemos si la mayoria de pixeles caen en el
# rango de color de barril.
# ==============================
# [2.2 v8.4] HSV mas laxo. En v8.3 con H_MAX=25 y S>=80 se perdian muchos
# pixeles del barril porque su naranja en Webots tira mas a 20-30 y la S
# baja en zonas en sombra/borde del barril.
BARREL_HSV_H_MIN              = 0      # antes 5
BARREL_HSV_H_MAX              = 30     # antes 25
BARREL_HSV_S_MIN              = 50     # antes 80
BARREL_HSV_V_MIN              = 30     # antes 40
# [2.2 v8.3] Sampling vertical en STRIP en lugar de parche fijo.
# El barril puede aparecer a distintas alturas segun la distancia, por eso
# muestreamos una franja vertical alta centrada en x_target.
BARREL_PATCH_W                = 40     # ancho del strip (antes 60)
BARREL_STRIP_Y_RATIO_TOP      = 0.45   # [2.2 v8.3] inicio vertical del strip (% del alto)
BARREL_STRIP_Y_RATIO_BOTTOM   = 0.95   # [2.2 v8.3] fin vertical del strip (% del alto)
BARREL_PIXEL_RATIO_THRESHOLD  = 0.05   # [2.2 v8.4] antes 0.10 - aun mas laxo


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
    # signal. Default 320 matches a 640-wide camera. Pass camera.getWidth()//2
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
        # of the chart - the old center line crossed it at 25 km/h, making 25
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
# YELLOW LINE DETECTION
# ==============================

def detect_yellow_line(frame):
    global yellow_line_prev

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

    # Brightness thresholds: keep only bright pixels (120-255).
    # The yellow road centerline in Webots is consistently bright (~180-255).
    # A lower bound of 120 excludes the dark road surface while still catching
    # the line under varying virtual lighting conditions.
    lower_yellow = 120
    upper_yellow = 255

    # cv2.inRange on a single-channel image with scalar bounds produces a
    # binary mask (255 where in range, 0 elsewhere) - same output type as the
    # HSV inRange used in V1.2SDF_, so all downstream steps are unchanged.
    yellow_mask = cv2.inRange(GRAY_, lower_yellow, upper_yellow)

    # ==============================
    # 2. NARROW ROI
    # Focus only where the centerline should appear
    # ==============================

    roi_mask = np.zeros_like(yellow_mask)

    # [2.2 v4] ROI vuelve a los valores originales de 2.1 (camara 640x320 ya escala bien)
    # En v3 se habia estrechado a (0.05 a 0.45) para evitar lineas blancas del lado derecho
    # pero perdiamos la linea al inicio. Con la camara ancha original esto deja de ser problema.
    vertices = np.array([[
        (int(width * 0.1), int(height * 1.00)),
        (int(width * 0.1), int(height * 0.55)),
        (int(width * 0.65), int(height * 0.75)),
        (int(width * 0.65), int(height * 1.00))
    ]], dtype=np.int32)

    cv2.fillPoly(roi_mask, vertices, 255)
    yellow_roi = cv2.bitwise_and(yellow_mask, roi_mask)

    # Clean mask with two-pass morphology.
    # OPEN (erode -> dilate) removes small isolated bright specks that the
    # grayscale threshold picks up more aggressively than the HSV mask did
    # (e.g. windshield glare, road texture highlights).
    # CLOSE (dilate -> erode) fills small gaps inside the line blob so
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

        # [2.2 v8.4] Guard contra division por cero / infinito cuando la pendiente
        # promedio cae casi a 0 (linea casi horizontal). Tratar como "linea no detectada".
        if not np.isfinite(avg_slope) or abs(avg_slope) < 1e-3:
            line_detected = False
            yellow_line_prev = None
            return line_detected, None, desired_x, debug_frame, yellow_roi

        line_x_reference = int((y_ref - avg_intercept) / avg_slope)

        # Reject impossible detections far from expected yellow-line area
        if not (width * 0.05 < line_x_reference < width * 0.55):
            line_detected = False
            yellow_line_prev = None
            return line_detected, None, desired_x, debug_frame, yellow_roi


        # Reject sudden jumps
        if yellow_line_prev is not None:
            prev_x = yellow_line_prev[0]

            if abs(line_x_reference - prev_x) > MAX_LINE_JUMP_PX:
                line_detected = False
                yellow_line_prev = None  # [2.2 v4] resetear prev para no quedar atrapado rechazando detecciones validas
                return line_detected, None, desired_x, debug_frame, yellow_roi

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

    return line_detected, line_x_reference, desired_x, debug_frame, yellow_roi


# ==============================
# [2.2 v5] LIDAR HELPER
# Lee los puntos centrales del SickLms291 y reporta el obstaculo mas cercano
# dentro de [LIDAR_MIN_VALID_M, LIDAR_MAX_DETECT_M].
#
# Retorna:
#   obstacle_detected (bool)
#   min_distance_m    (float, inf si no hay nada)
#   idx_center_offset (int, offset en puntos respecto al centro del scan)
#                     negativo = izquierda, positivo = derecha
# ==============================
def read_lidar_front(lidar):
    ranges = lidar.getRangeImage()
    if ranges is None or len(ranges) == 0:
        return False, float('inf'), 0

    n_total   = len(ranges)
    total_fov_deg = np.degrees(lidar.getFov())  # 180 por defecto
    if total_fov_deg <= 0:
        return False, float('inf'), 0

    # Cuantos puntos cubren los LIDAR_FRONT_FOV_DEG centrales
    n_front = max(1, int(n_total * (LIDAR_FRONT_FOV_DEG / total_fov_deg)))
    center  = n_total // 2
    half    = n_front // 2
    i_start = max(0, center - half)
    i_end   = min(n_total, center + half + 1)

    window = np.array(ranges[i_start:i_end], dtype=np.float32)

    # Filtrar lecturas invalidas (inf, NaN, fuera de rango util)
    valid_mask = (
        np.isfinite(window)
        & (window >= LIDAR_MIN_VALID_M)
        & (window <= LIDAR_MAX_DETECT_M)
    )

    if not np.any(valid_mask):
        return False, float('inf'), 0

    min_idx_local  = int(np.argmin(np.where(valid_mask, window, np.inf)))
    min_distance_m = float(window[min_idx_local])
    idx_global     = i_start + min_idx_local
    idx_offset     = idx_global - center

    return True, min_distance_m, idx_offset


# ==============================
# [2.2 v6] LIDAR-TO-CAMERA MAPPING
# El SickLms291 cubre 180 grados con 180 puntos.
# La camara tiene fieldOfView=1 rad (~57.3 grados) en el .wbt.
# offset_pts del LiDAR (negativo=izq, positivo=der) se proyecta
# a una columna de pixel en la imagen de la camara.
# ==============================
def lidar_offset_to_camera_x(offset_pts, lidar_fov_deg, cam_fov_deg, cam_width):
    pts_per_deg = max(1.0, 180.0 / max(1.0, lidar_fov_deg))  # 180 puntos / 180 deg = 1
    offset_deg  = offset_pts / pts_per_deg
    # Si el obstaculo cae fuera del FOV de la camara devolvemos None
    if abs(offset_deg) > cam_fov_deg / 2.0:
        return None
    col = cam_width / 2.0 + (offset_deg / cam_fov_deg) * cam_width
    return int(np.clip(col, 0, cam_width - 1))


# ==============================
# [2.2 v6] SVM SLIDING WINDOW DETECTOR
# Recorta una franja vertical en torno a x_center_px y corre sliding window
# multi-escala. Para cada ventana 64x128 calcula HOG y predice con LinearSVC.
# Retorna lista de detecciones positivas (x, y, w, h, score).
# ==============================
def detect_pedestrian_window(frame_gray, x_center_px, svm_model):
    if svm_model is None or frame_gray is None:
        return []

    H, W = frame_gray.shape[:2]
    x_min = max(0, int(x_center_px - SVM_ROI_HALF_WIDTH_PX))
    x_max = min(W, int(x_center_px + SVM_ROI_HALF_WIDTH_PX))
    strip = frame_gray[:, x_min:x_max]
    Hs, Ws = strip.shape[:2]

    detections = []

    # [2.2 v8.4] Salto la parte superior del frame (cielo/edificios)
    y_start_min = int(Hs * SVM_Y_START_MIN_RATIO)

    for scale in SVM_SCALES:
        win_w = int(SVM_WIN_W * scale)
        win_h = int(SVM_WIN_H * scale)
        if win_w >= Ws or win_h >= Hs:
            continue

        for y in range(y_start_min, Hs - win_h + 1, SVM_STEP_Y):
            for x in range(0, Ws - win_w + 1, SVM_STEP_X):
                patch = strip[y:y + win_h, x:x + win_w]
                # Normalizar siempre a 64x128 antes de HOG
                if patch.shape[:2] != (SVM_WIN_H, SVM_WIN_W):
                    patch = cv2.resize(patch, (SVM_WIN_W, SVM_WIN_H))
                try:
                    feats = hog(patch, **SVM_HOG_PARAMS)
                    pred  = int(svm_model.predict([feats])[0])
                except Exception:
                    continue
                if pred == 1:
                    # Score por decision_function (margen del SVM)
                    try:
                        score = float(svm_model.decision_function([feats])[0])
                    except Exception:
                        score = 1.0
                    # [2.2 v6.1] Filtrar detecciones debiles
                    if score < SVM_SCORE_THRESHOLD:
                        continue
                    # Coordenadas relativas al frame completo
                    detections.append((x_min + x, y, win_w, win_h, score))

    return detections


# ==============================
# [2.2 v6.1] NON-MAXIMUM SUPPRESSION
# Consolida cajas superpuestas en una sola por objeto. Usa cv2.dnn.NMSBoxes.
# Entrada: lista de (x, y, w, h, score)
# Salida : lista reducida con la misma forma
# ==============================
def apply_nms(detections, iou_threshold=SVM_NMS_IOU, score_threshold=SVM_SCORE_THRESHOLD):
    if not detections:
        return []
    boxes  = [[int(x), int(y), int(w), int(h)] for (x, y, w, h, _) in detections]
    scores = [float(s) for (_, _, _, _, s) in detections]
    try:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)
    except Exception:
        return detections
    if idxs is None or len(idxs) == 0:
        return []
    # cv2.dnn.NMSBoxes puede devolver array Nx1 o lista plana segun version
    flat = [int(i[0]) if hasattr(i, '__iter__') else int(i) for i in idxs]
    return [detections[i] for i in flat]


# ==============================
# [2.2 v8] DISCRIMINADOR POR COLOR PARA BARRIL
# Toma un parche del frame en torno a x_target_px y devuelve:
#   (is_barrel: bool, color_ratio: float)
# color_ratio = fraccion de pixeles del parche que caen en el rango naranja-oxido del barril.
# ==============================
def detect_barrel_color(frame, x_target_px):
    """
    [2.2 v8.4] Devuelve: (is_barrel, ratio, hsv_mean, bbox, breakdown)
      Muestrea un STRIP vertical ALTO centrado en x_target_px.
      breakdown = dict con % de pixeles que pasan cada condicion individual
                  (h_in, s_in, v_in) y la combinada (all_in == ratio).
                  Util para diagnosticar cual umbral nos esta restringiendo.
    """
    if frame is None or x_target_px is None:
        return False, 0.0, (0, 0, 0), None, {}
    H, W = frame.shape[:2]
    half_w = BARREL_PATCH_W // 2
    x_min = max(0, int(x_target_px) - half_w)
    x_max = min(W, int(x_target_px) + half_w)
    y_min = max(0, int(H * BARREL_STRIP_Y_RATIO_TOP))
    y_max = min(H, int(H * BARREL_STRIP_Y_RATIO_BOTTOM))
    strip = frame[y_min:y_max, x_min:x_max]
    bbox = (x_min, y_min, x_max, y_max)
    if strip.size == 0:
        return False, 0.0, (0, 0, 0), bbox, {}
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    H_ch = hsv[:, :, 0]
    S_ch = hsv[:, :, 1]
    V_ch = hsv[:, :, 2]
    mask_h = (H_ch >= BARREL_HSV_H_MIN) & (H_ch <= BARREL_HSV_H_MAX)
    mask_s = (S_ch >= BARREL_HSV_S_MIN)
    mask_v = (V_ch >= BARREL_HSV_V_MIN)
    mask   = mask_h & mask_s & mask_v
    ratio  = float(mask.mean())
    is_barrel = ratio >= BARREL_PIXEL_RATIO_THRESHOLD
    if mask.any():
        hsv_mean = (int(H_ch[mask].mean()), int(S_ch[mask].mean()), int(V_ch[mask].mean()))
    else:
        hsv_mean = (int(H_ch.mean()), int(S_ch.mean()), int(V_ch.mean()))
    breakdown = {
        "h_in": float(mask_h.mean()),
        "s_in": float(mask_s.mean()),
        "v_in": float(mask_v.mean()),
        "all_in": ratio,
    }
    return is_barrel, ratio, hsv_mean, bbox, breakdown


# ==============================
# [2.2 v7.1] FILTRO GEOMETRICO PEATON
# Toma detecciones del SVM y devuelve solo las que tienen forma plausible
# de peaton: bbox alto, top dentro del tercio superior/medio de la imagen.
# Las que no cumplen probablemente son barril/poste/sombra.
# ==============================
def filter_pedestrian_geometry(detections, frame_height):
    out = []
    max_top_y = frame_height * STATE_PED_MAX_TOP_Y_RATIO
    for (x, y, w, h, s) in detections:
        if h < STATE_PED_MIN_BBOX_HEIGHT_PX:
            continue
        if y > max_top_y:
            continue
        out.append((x, y, w, h, s))
    return out


# ==============================
# MAIN CONTROLLER
# ==============================

def main():
    speed = 0
    angle = 0.0
    previous_angle = 0.0   # seed value for the steering smoothing
    last_press = {}

    # [2.2 TEMP] Arrancar directamente en modo autonomo para probar el seguidor
    # de linea sin entradas manuales. Las secciones de teclado y PS4 estan
    # comentadas mas abajo. Para regresar a manual: cambiar a False y
    # destapar los bloques de input.
    autonomous_mode = True

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

    # [2.2 v5] LiDAR Sick LMS 291 montado en sensorsSlotFront del BmwX5
    lidar = robot.getDevice("Sick LMS 291")
    if lidar is not None:
        lidar.enable(timestep)
        try:
            print(
                f"[LIDAR] device='Sick LMS 291' "
                f"horizRes={lidar.getHorizontalResolution()} "
                f"fov={np.degrees(lidar.getFov()):.1f}deg "
                f"maxRange={lidar.getMaxRange():.1f}m"
            )
        except Exception as e:
            print(f"[LIDAR] info no disponible: {e}")
    else:
        print("[LIDAR] WARN: no se encontro device 'Sick LMS 291'")

    # [2.2 v6] Cargar el SVM entrenado para detectar peatones
    svm_model = None
    try:
        svm_model = joblib.load(SVM_MODEL_PATH)
        print(f"[SVM] modelo cargado de {SVM_MODEL_PATH}")
    except Exception as e:
        print(f"[SVM] WARN: no se pudo cargar {SVM_MODEL_PATH}: {e}")

    # [2.2 v6] Contador de frames para hacer throttle del SVM
    svm_frame_counter = 0
    last_svm_detections = []  # ultimas detecciones para dibujar entre corridas del SVM

    # [2.2 v7] Estado de la maquina de obstaculos
    obstacle_state    = "NORMAL"   # NORMAL | PEDESTRIAN_STOP | BARREL_STOP
    svm_hits_streak   = 0           # scans consecutivos con peaton detectado
    svm_misses_streak = 0           # scans consecutivos sin peaton pero con LiDAR cerca
    stop_entered_time = None        # [2.2 v7.3] cuando entramos al estado STOP
    bypass_bias_angle = 0.0         # [2.2 v7.4] sesgo angular activo para bordear obstaculo
    bypass_bias_until = 0.0         # [2.2 v7.4] timestamp hasta el cual el sesgo se aplica
    last_color_bbox      = None     # [2.2 v8.1] bbox del ultimo parche de color para overlay
    last_color_is_barrel = False    # [2.2 v8.1] resultado del ultimo color check

    display_img = Display("display_image")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    # [2.2 TEMP] PS4 deshabilitado: queremos modo automatico sin entradas.
    # joystick = init_ps_controller()
    joystick = None

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
    print("PS4 X = toggle autonomous mode")
    print("Square = save image")

    while robot.step() != -1:
        current_time = time.time()
        dt = current_time - previous_time
        previous_time = current_time

        frame = get_image(camera)

        # Default display (overridden at end of loop by the debug frame resize)
        display_frame = cv2.resize(frame, (640, 300))

        # ==============================
        # [2.2 v5] LIDAR READ
        # Solo lectura + reporte. NO frena el carro todavia.
        # La maquina de estados (tarea 7) decide que hacer.
        # ==============================
        lidar_obstacle = False
        lidar_dist_m   = float('inf')
        lidar_offset   = 0
        if lidar is not None:
            lidar_obstacle, lidar_dist_m, lidar_offset = read_lidar_front(lidar)
            if lidar_obstacle:
                print(
                    f"[LIDAR] OBSTACLE | dist={lidar_dist_m:5.2f}m "
                    f"| offset_pts={lidar_offset:+d}"
                )

        # ==============================
        # [2.2 v6] SVM SLIDING WINDOW
        # Solo correr el SVM cuando el LiDAR vio algo, con throttle de N frames
        # para no matar el framerate.
        # ==============================
        svm_frame_counter += 1
        run_svm_now = (
            svm_model is not None
            and lidar_obstacle
            and (svm_frame_counter % SVM_RUN_EVERY_N_FRAMES == 0)
        )
        if run_svm_now:
            cam_fov_deg   = np.degrees(camera.getFov()) if camera.getFov() > 0 else 57.3
            lidar_fov_deg = np.degrees(lidar.getFov())  if lidar.getFov()  > 0 else 180.0
            x_target = lidar_offset_to_camera_x(
                lidar_offset, lidar_fov_deg, cam_fov_deg, frame.shape[1]
            )
            if x_target is not None:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                raw_detections = detect_pedestrian_window(
                    gray_frame, x_target, svm_model
                )
                # [2.2 v6.1] Aplicar NMS para colapsar cajas superpuestas
                last_svm_detections = apply_nms(raw_detections)
                # [2.2 v7.1] Filtrar geometricamente: solo peatones plausibles cuentan
                pedestrian_detections = filter_pedestrian_geometry(
                    last_svm_detections, frame.shape[0]
                )
                if pedestrian_detections:
                    top = max(pedestrian_detections, key=lambda d: d[4])
                    print(
                        f"[SVM] PEDESTRIAN | n={len(pedestrian_detections)} "
                        f"| x_target={x_target} | top_score={top[4]:+.3f}"
                    )
                else:
                    if last_svm_detections:
                        print(f"[SVM] descartado por geometria (no peaton) | x_target={x_target} | n_raw={len(last_svm_detections)}")
                    else:
                        print(f"[SVM] no peaton | x_target={x_target}")
            else:
                last_svm_detections = []
        elif not lidar_obstacle:
            last_svm_detections = []   # limpia overlay cuando ya no hay obstaculo

        # ==============================
        # [2.2 v7] OBSTACLE STATE MACHINE
        # Actualizar contadores y transicionar entre NORMAL / PEDESTRIAN_STOP / BARREL_STOP
        # ==============================

        # 3a) Actualizar streaks SOLO en los frames donde el SVM realmente corrio.
        # [2.2 v7.1] Usar las detecciones FILTRADAS por geometria, no las crudas.
        # Esto evita que el barril dispare svm_hits_streak (es vertical pero corto).
        if run_svm_now and lidar_obstacle:
            pedestrian_now = filter_pedestrian_geometry(
                last_svm_detections, frame.shape[0]
            )
            if pedestrian_now:
                svm_hits_streak  += 1
                svm_misses_streak = 0
            else:
                svm_misses_streak += 1
                svm_hits_streak    = 0

        # 3b) Cuando el LiDAR no ve nada, resetear todo
        if not lidar_obstacle:
            svm_hits_streak   = 0
            svm_misses_streak = 0

        # 3c) Transiciones de estado con histeresis
        new_state = obstacle_state

        # [2.2 v8.2] Correr color check SIEMPRE que el LiDAR vea obstaculo cerca,
        # no solo en la transicion NORMAL->STOP. Asi el estado puede cambiar entre
        # PEDESTRIAN_STOP y BARREL_STOP segun lo que este enfrente en este momento.
        current_is_barrel = None  # None = no se hizo color check este frame
        if lidar_obstacle and lidar_dist_m <= STATE_BRAKE_DIST_M:
            cam_fov_deg   = np.degrees(camera.getFov()) if camera.getFov() > 0 else 57.3
            lidar_fov_deg = np.degrees(lidar.getFov())  if (lidar and lidar.getFov() > 0) else 180.0
            x_target_state = lidar_offset_to_camera_x(
                lidar_offset, lidar_fov_deg, cam_fov_deg, frame.shape[1]
            )
            if x_target_state is not None:
                # [2.2 v8.4] detect_barrel_color ahora devuelve 5 valores (con breakdown)
                is_barrel, color_ratio, hsv_mean, color_bbox, color_breakdown = detect_barrel_color(
                    frame, x_target_state
                )
                h_m, s_m, v_m = hsv_mean
                # Print cada N frames para no saturar la terminal
                if svm_frame_counter % 5 == 0:
                    bd = color_breakdown
                    print(f"[COLOR] offset={lidar_offset:+d} x_target={x_target_state} "
                           f"| HSV_mean=({h_m:3d},{s_m:3d},{v_m:3d}) "
                           f"| ratio={color_ratio:.3f} "
                           f"| h_in={bd.get('h_in',0):.2f} s_in={bd.get('s_in',0):.2f} v_in={bd.get('v_in',0):.2f} "
                           f"| is_barrel={is_barrel}")
                last_color_bbox      = color_bbox
                last_color_is_barrel = is_barrel
                current_is_barrel    = is_barrel

        # [2.2 v8.3] Maquina de estados reestructurada.
        # Bug en v8.2: el else con la logica de timeout NUNCA corria porque
        # el if/elif ya cubria los 3 estados posibles. Ahora la logica de
        # cambio de sub-estado (PED <-> BAR) y la de timeout/resume corren
        # AMBAS cuando estamos en cualquier estado de STOP.
        if obstacle_state == "NORMAL":
            if current_is_barrel is True:
                new_state = "BARREL_STOP"
            elif current_is_barrel is False:
                new_state = "PEDESTRIAN_STOP"
        else:
            # Estamos en PEDESTRIAN_STOP o BARREL_STOP
            # 1) Cambio de sub-estado segun color del frame actual
            if obstacle_state == "PEDESTRIAN_STOP" and current_is_barrel is True:
                new_state = "BARREL_STOP"
                stop_entered_time = current_time  # reset timer al cambiar de sub-estado
            elif obstacle_state == "BARREL_STOP" and current_is_barrel is False:
                new_state = "PEDESTRIAN_STOP"
                stop_entered_time = current_time

            # 2) Resume / timeout (SIEMPRE evaluar mientras estemos parados)
            time_in_stop = (current_time - stop_entered_time) if stop_entered_time else 0.0
            if time_in_stop >= STATE_MIN_STOP_DURATION_S:
                if (not lidar_obstacle) or (lidar_dist_m >= STATE_RESUME_DIST_M):
                    new_state = "NORMAL"
                elif time_in_stop >= STATE_MAX_STOP_DURATION_S:
                    # [2.2 v8.3] Emergency bypass: si llevamos MAX_STOP_DURATION_S
                    # sin que el LiDAR se despeje, asumimos obstaculo permanente.
                    # Forzar salida con maniobra de bordeo angular.
                    new_state = "NORMAL"
                    if lidar_offset > 0:
                        bypass_bias_angle = -BYPASS_ANGLE_MAGNITUDE  # obstaculo a la der -> sesgo izq
                    elif lidar_offset < 0:
                        bypass_bias_angle = +BYPASS_ANGLE_MAGNITUDE  # obstaculo a la izq -> sesgo der
                    else:
                        bypass_bias_angle = +BYPASS_ANGLE_MAGNITUDE  # default: bordeo a la der
                    bypass_bias_until = current_time + BYPASS_DURATION_S
                    print(f"[BYPASS] forzando salida tras {time_in_stop:.1f}s (de {obstacle_state}) | "
                          f"lidar_offset={lidar_offset:+d} | bias_angle={bypass_bias_angle:+.2f} rad "
                          f"| duracion={BYPASS_DURATION_S}s")

        if new_state != obstacle_state:
            print(f"[STATE] {obstacle_state} -> {new_state} | dist={lidar_dist_m:.2f}m "
                   f"| svm_hits={svm_hits_streak} | svm_misses={svm_misses_streak}")
            obstacle_state = new_state
            # [2.2 v7.3] marcar el momento de entrada a STOP para el timer
            if new_state in ("PEDESTRIAN_STOP", "BARREL_STOP"):
                stop_entered_time = current_time
            else:
                stop_entered_time = None

        # ==============================
        # [2.2 TEMP] KEYBOARD CONTROL - deshabilitado para pruebas en automatico
        # Para reactivar, elimina las triple comillas (""") al inicio y al final.
        # ==============================
        """
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
        """

        # ==============================
        # [2.2 TEMP] PS4 CONTROLLER MANUAL MODE - deshabilitado para automatico
        # Para reactivar, elimina las triple comillas (""") al inicio y al final.
        # ==============================
        """
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
        """

        # ==============================
        # ALWAYS RUN YELLOW LINE DETECTION
        # This allows debugging in MANUAL and AUTO
        # ==============================

        line_detected, line_x, desired_x, debug_frame, yellow_roi = detect_yellow_line(frame)

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
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,   # [2.2] antes 0.7 - achicado para que quepa en la ventana
                (0, 255, 255),
                1      # [2.2] antes 2 - grosor reducido por mismo motivo
            )


        else:
            error = 0

            cv2.putText(
                debug_frame,
                "Target: keep car centered in right lane using yellow line",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,   # [2.2] antes 0.7 - achicado
                (255, 0, 255),
                1      # [2.2] antes 2 - grosor reducido
            )


        # ==============================
        # AUTONOMOUS MODE USES PID
        # ==============================

        if autonomous_mode:

            if line_detected:
                # Line is visible - clear the fallback timer immediately so the
                # countdown only runs during continuous loss, not accumulated loss.
                line_lost_time = None

                # error already computed above; no need to recalculate
                pid_output = pid.compute(error, dt)

                # Steering smoothing: blends previous angle with new PID output
                # to reduce abrupt steering changes between frames.
                #angle = 0.8 * previous_angle + 0.2 * pid_output
                angle = 0.6 * previous_angle + 0.4 * pid_output

                # Save for next frame
                previous_angle = angle

                # Speed adaptation: reduce speed proportionally when error is
                # large (sharp turn ahead); speed up when driving straight.
                speed = AUTONOMOUS_SPEED - min(abs(error) * 0.15, 10)
                speed = max(speed, 8)

                cv2.putText(
                    debug_frame,
                    f"AUTO | PID angle: {angle:.3f}",
                    (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,   # [2.2] antes 0.8 - achicado
                    (0, 255, 255),
                    1      # [2.2] antes 2 - grosor reducido
                )

                # [2.2] Print diagnostico a la terminal para tunear el seguidor de linea
                print(
                    f"[AUTO] line_x={line_x:>3} | lane_c={lane_center_x:>3} "
                    f"| err={error:>4} | pid_out={pid_output:+.3f} "
                    f"| angle={angle:+.3f} | spd={speed:.0f}"
                )

            else:
                speed = 10
                angle = 0.0

                # Start the fallback timer the first frame the line goes missing.
                if line_lost_time is None:
                    line_lost_time = current_time

                # Compute remaining seconds (ceiling so display shows 5->4->...->1->0).
                time_remaining = max(0.0, 5.0 - (current_time - line_lost_time))
                countdown = int(np.ceil(time_remaining))

                cv2.putText(
                    debug_frame,
                    "AUTO | Searching yellow line",
                    (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,   # [2.2] antes 0.8 - achicado
                    (0, 0, 255),
                    1      # [2.2] antes 2 - grosor reducido
                )

                # Countdown shown below the searching message.
                # Color shifts from orange -> red as time runs out.
                countdown_color = (0, 100, 255) if countdown > 2 else (0, 0, 255)
                cv2.putText(
                    debug_frame,
                    f"Going back to Manual MODE: {countdown}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,   # [2.2] antes 0.7 - achicado
                    countdown_color,
                    1      # [2.2] antes 2 - grosor reducido
                )

                # [2.2] Print diagnostico cuando la linea se pierde
                print(f"[AUTO] LINE LOST | angle=0.000 | spd=10 | countdown={countdown}s")

                # [2.2 TEMP] Auto-revert deshabilitado para mantener el modo
                # automatico durante todas las pruebas, incluso si la linea se
                # pierde unos segundos. Para reactivar, destapa las lineas.
                # if time_remaining <= 0:
                #     autonomous_mode = False
                #     line_lost_time = None
                #     pid.reset()
                #     previous_angle = 0.0
                #     print("Line lost - automatically reverted to MANUAL mode")

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

        # Speed overlay - drawn on every frame in both manual and autonomous
        # modes so the operator always has current speed visible in the main
        # debug window without needing to read the separate PID chart.
        speed_color = (0, 255, 255) if autonomous_mode else (0, 255, 0)
        cv2.putText(
            debug_frame,
            f"Speed: {speed:.0f} km/h",
            (10, debug_frame.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            speed_color,
            1
        )

        # [2.2 v8.1] Dibujar el parche de color (caja blanca/naranja) si hubo color check reciente
        if last_color_bbox is not None:
            (cb_x1, cb_y1, cb_x2, cb_y2) = last_color_bbox
            box_color = (0, 140, 255) if last_color_is_barrel else (255, 255, 255)
            cv2.rectangle(debug_frame, (cb_x1, cb_y1), (cb_x2, cb_y2), box_color, 2)
            label = "BARREL" if last_color_is_barrel else "NO-BAR"
            cv2.putText(debug_frame, label, (cb_x1, max(10, cb_y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        # [2.2 v7] Overlay del estado de la maquina de obstaculos (esquina superior derecha)
        if obstacle_state == "PEDESTRIAN_STOP":
            state_text  = "PEDESTRIAN - STOP"
            state_color = (0, 0, 255)        # rojo
        elif obstacle_state == "BARREL_STOP":
            state_text  = "BARREL - STOP + HAZARDS"
            state_color = (0, 140, 255)      # naranja
        else:
            state_text  = "NORMAL"
            state_color = (0, 255, 0)        # verde
        cv2.putText(
            debug_frame,
            state_text,
            (debug_frame.shape[1] - 230, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            state_color,
            2
        )

        # [2.2 v6] Dibujar bounding boxes del SVM peaton (verdes)
        for (bx, by, bw, bh, bscore) in last_svm_detections:
            cv2.rectangle(
                debug_frame,
                (int(bx), int(by)),
                (int(bx + bw), int(by + bh)),
                (0, 255, 0),
                2
            )
            cv2.putText(
                debug_frame,
                f"P {bscore:+.2f}",
                (int(bx), max(10, int(by) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        # [2.2 v5] Overlay del LiDAR en la esquina inferior derecha del debug
        if lidar_obstacle:
            lidar_text  = f"LiDAR: {lidar_dist_m:4.1f}m [!]"
            lidar_color = (0, 0, 255)   # rojo cuando hay obstaculo en el FOV frontal
        else:
            lidar_text  = "LiDAR: clear"
            lidar_color = (0, 255, 0)   # verde cuando esta libre
        cv2.putText(
            debug_frame,
            lidar_text,
            (debug_frame.shape[1] - 170, debug_frame.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            lidar_color,
            1
        )

        # ==============================
        # [2.2 v7.2] STATE ACTION + SEND COMMANDS (CONSOLIDADO)
        #
        # En Webots, llamar setCruisingSpeed() RE-ACTIVA el cruise control
        # y libera el freno, sin importar que setBrakeIntensity(1.0) se haya
        # llamado antes. Por eso v7 y v7.1 no frenaban.
        #
        # Fix: cuando estamos en estado STOP NO llamamos setCruisingSpeed.
        # Solo aplicamos freno + corta acelerador.
        # ==============================
        if obstacle_state in ("PEDESTRIAN_STOP", "BARREL_STOP"):
            speed = 0
            angle = 0.0
            driver.setSteeringAngle(0.0)
            # [2.2 v7.3] combinacion COMPLETA de comandos de freno de emergencia:
            #   throttle=0  -> nada de acelerador
            #   cruise=0    -> objetivo de velocidad cero
            #   brake=1.0   -> freno maximo
            # El brake intensity domina sobre el cruise control.
            brake_calls_ok = True
            try:
                driver.setThrottle(0.0)
            except Exception as e:
                brake_calls_ok = False
                print(f"[BRAKE] setThrottle FAILED: {e}")
            try:
                driver.setCruisingSpeed(0.0)
            except Exception as e:
                brake_calls_ok = False
                print(f"[BRAKE] setCruisingSpeed FAILED: {e}")
            try:
                driver.setBrakeIntensity(STATE_BRAKE_INTENSITY)
            except Exception as e:
                brake_calls_ok = False
                print(f"[BRAKE] setBrakeIntensity FAILED: {e}")

            # Hazard flashers solo para el barril
            try:
                if obstacle_state == "BARREL_STOP":
                    driver.setHazardFlashers(True)
                else:
                    driver.setHazardFlashers(False)
            except Exception:
                pass

            # Print diagnostico 1 de cada 10 frames para confirmar que se aplica
            if brake_calls_ok and (svm_frame_counter % 10 == 0):
                print(f"[BRAKE] state={obstacle_state} | brake={STATE_BRAKE_INTENSITY} | throttle=0 | cruise=0")

        else:
            # NORMAL: liberar freno, apagar hazards, usar cruise control normal
            try:
                driver.setBrakeIntensity(0.0)
            except Exception:
                pass
            try:
                driver.setHazardFlashers(False)
            except Exception:
                pass
            # [2.2 v7.4] Aplicar sesgo de bypass si esta activo (justo despues
            # de salir de un STOP forzado por timeout).
            if current_time < bypass_bias_until:
                angle = float(np.clip(angle + bypass_bias_angle, -MAX_ANGLE, MAX_ANGLE))
                # speed reducida durante la maniobra de bordeo
                speed = min(speed, 25)
                if svm_frame_counter % 10 == 0:
                    remaining = bypass_bias_until - current_time
                    print(f"[BYPASS] activo | angle={angle:+.3f} | speed={speed} | restante={remaining:.1f}s")
            driver.setSteeringAngle(angle)
            driver.setCruisingSpeed(speed)

        # [2.2 v7.2] Redibujar el overlay de Speed con el valor POST-override
        # (tapamos el overlay viejo con un rectangulo negro y dibujamos encima)
        cv2.rectangle(
            debug_frame,
            (5,  debug_frame.shape[0] - 22),
            (170, debug_frame.shape[0] - 2),
            (0, 0, 0),
            -1
        )
        post_speed_color = (0, 0, 255) if obstacle_state != "NORMAL" else (
            (0, 255, 255) if autonomous_mode else (0, 255, 0)
        )
        cv2.putText(
            debug_frame,
            f"Speed: {speed:.0f} km/h",
            (10, debug_frame.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            post_speed_color,
            1
        )

        # [2.2 v7.2] display_frame se construye al FINAL, ya con todos los
        # overlays (incluido el speed post-override) dibujados sobre debug_frame.
        display_frame = cv2.resize(debug_frame, (640, 300))

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
