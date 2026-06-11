# ============================================================
# dataset_mode_v3 — Dataset Capture Module (V1.8 controller)
# ============================================================
#
# PURPOSE
#   Auto-capture Webots camera frames at a fixed rate and produce
#   YOLO-format bounding-box labels using color + shape heuristics.
#   Intended for training a YOLO traffic sign / traffic light detector.
#
# TRIGGER
#   PS4 Triangle (btn 2) or Keyboard 'D' / 'd' → toggle on/off.
#   Programmatic: set_dataset_mode(True/False).
#
# OUTPUT
#   data/webots_dataset/images/frame_XXXXXX.jpg
#   data/webots_dataset/labels/frame_XXXXXX.txt  (YOLO: cls cx cy w h, normalized)
#   Empty .txt = hard-negative sample (no objects detected, still useful for training).
#
# CAPTURE RATE
#   One frame saved every 2 seconds while dataset mode is active.
#
# CLASS IDs  (6 classes)
#   0  stop            — red octagon / red near-square sign
#   1  speed_limit     — red-rimmed circular sign
#   2  priority_warning— yellow triangle warning sign or blue circle priority sign
#   3  traffic_light_red
#   4  traffic_light_yellow
#   5  traffic_light_green
#
# LABELING PIPELINE  (per frame)
#   Frame is split into a top zone (top 30 % of height, traffic lights)
#   and a lower zone (road signs).
#
#   RED channel
#     Top zone  → traffic_light_red (class 3), blob centroid + radius box.
#     Lower zone → contour-level shape analysis:
#       • 4+ vertices, circularity 0.50–0.90, aspect 0.70–1.40 → stop (0)
#       • circularity > 0.72 → speed_limit (1)
#       • fallback: area ≥ 2500 px → stop (0), else speed_limit (1)
#
#   YELLOW channel  (HSV 18–35°)
#     Top zone  → traffic_light_yellow (class 4).
#     Lower zone → priority_warning (class 2) for triangle warning signs.
#
#   GREEN channel  (two HSV bands: 35–70° and 75–90°)
#     Top zone only → traffic_light_green (class 5).
#
#   BLUE channel  (HSV 100–130°)
#     Lower zone → priority_warning (class 2) for blue circle priority signs.
#
# POST-PROCESSING
#   Deduplication: IoU-based NMS per class (threshold 0.6), keeps largest box.
#   Edge filter: drops boxes touching frame bottom (y > 0.95) or side edges
#                (x < 0.03 or x > 0.97) to eliminate ground reflections and
#                partially visible signs.
#
# DEBUG OVERLAY
#   Returns an annotated BGR copy of the frame with colored bounding boxes
#   and class name labels; counter badge in top-right corner.
# ============================================================

import cv2
import math
import numpy as np
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
dataset_mode = False
dataset_counter = 0
last_capture_time = 0.0

# ---------------------------------------------------------------------------
# Class ID mapping (YOLO format)
# 0: Stop (red octagon / red circle with text)
# 1: Speed Limit (red-rimmed circle with number)
# 2: Priority / Warning (yellow triangle, inverted triangle)
# 3: Traffic Light Red
# 4: Traffic Light Yellow
# 5: Traffic Light Green
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "stop",
    1: "speed_limit",
    2: "priority_warning",
    3: "traffic_light_red",
    4: "traffic_light_yellow",
    5: "traffic_light_green",
}

CLASS_COLORS = {
    0: (255, 0, 0),    # Stop — red
    1: (255, 165, 0),  # Speed — orange
    2: (0, 255, 255),  # Priority — yellow
    3: (255, 0, 0),    # Red light
    4: (255, 255, 0),  # Yellow light
    5: (0, 255, 0),    # Green light
}

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "webots_dataset")
IMG_DIR = os.path.join(DATASET_DIR, "images")
LAB_DIR = os.path.join(DATASET_DIR, "labels")


# ---------------------------------------------------------------------------
# Toggle
# ---------------------------------------------------------------------------

def toggle_dataset():
    global dataset_mode
    dataset_mode = not dataset_mode
    if dataset_mode:
        ensure_dirs()
        print(f"\n{'='*50}")
        print(f"  DATASET MODE ON  |  {dataset_counter} frames captured")
        print(f"  Save dir: {IMG_DIR}")
        print(f"  Capture every: 2s  |  Trigger: Triangle / D")
        print(f"{'='*50}\n")
    else:
        print(f"  [DATASET MODE OFF] Total frames captured: {dataset_counter}")
        print()


def set_dataset_mode(enabled: bool):
    global dataset_mode
    dataset_mode = enabled
    if enabled:
        ensure_dirs()
        print(f"\n{'='*50}")
        print(f"  DATASET MODE ON  |  {dataset_counter} frames captured")
        print(f"  Save dir: {IMG_DIR}")
        print(f"  Capture every: 2s  |  Trigger: Triangle / D")
        print(f"{'='*50}\n")
    else:
        print(f"  [DATASET MODE OFF] Total frames captured: {dataset_counter}")
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LAB_DIR, exist_ok=True)


def _find_blob_centroids(mask, min_area=200, max_area=80000):
    """Return list of (cx, cy, area) for contours in mask."""
    centers = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy, area))
    return centers


def _box_iou(b1, b2):
    """
    b1, b2 in YOLO normalized: (cx, cy, w, h)
    Returns intersection-over-union.
    """
    x1a = b1[0] - b1[2] / 2
    y1a = b1[1] - b1[3] / 2
    x2a = x1a + b1[2]
    y2a = y1a + b1[3]

    x1b = b2[0] - b2[2] / 2
    y1b = b2[1] - b2[3] / 2
    x2b = x1b + b2[2]
    y2b = y1b + b2[3]

    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    iw = max(0, xi2 - xi1)
    ih = max(0, yi2 - yi1)
    inter = iw * ih

    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _dedup_labels(labels, iou_thresh=0.6):
    """Remove overlapping boxes that share the same class, keep the biggest."""
    if not labels:
        return labels

    # Sort by area (descending) so we keep the largest first
    with_areas = []
    for cls, xc, yc, wn, hn in labels:
        area = wn * hn
        with_areas.append((cls, xc, yc, wn, hn, area))
    with_areas.sort(key=lambda x: x[5], reverse=True)

    keep = []
    for entry in with_areas:
        cls, xc, yc, wn, hn, area = entry
        dominated = False
        for k in keep:
            k_cls, k_xc, k_yc, k_wn, k_hn, k_area = k
            if cls == k_cls and _box_iou((xc, yc, wn, hn), (k_xc, k_yc, k_wn, k_hn)) > iou_thresh:
                dominated = True
                break
        if not dominated:
            keep.append(entry)

    return [(cls, xc, yc, wn, hn) for cls, xc, yc, wn, hn, _ in keep]


def _filter_edge_detections(labels, frame_h, frame_w):
    """Discard detections too close to image borders or bottom ground."""
    filtered = []
    for cls, xc, yc, wn, hn in labels:
        # Distance from bottom (ground reflections)
        bottom_edge = yc + hn / 2
        if bottom_edge > 0.95:
            continue
        # Distance from left/right edges
        if xc - wn / 2 < 0.03 or xc + wn / 2 > 0.97:
            continue
        filtered.append((cls, xc, yc, wn, hn))
    return filtered


# ---------------------------------------------------------------------------
# Labeling — color/shape heuristics
# ---------------------------------------------------------------------------

def label_frame(frame):
    """
    Analyze one camera frame, assign YOLO bounding-box labels.
    Returns: list of (class_id, x_center, y_center, width, height) normalized.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    labels = []

    # ------------------------------------------------------------------
    # 1. RED objects (traffic light red, stop signs, speed limit rims)
    # ------------------------------------------------------------------
    red1 = cv2.inRange(hsv, (0, 120, 50), (15, 255, 255))
    red2 = cv2.inRange(hsv, (165, 120, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # Traffic lights are in the top ~30% of frame
    top_h = int(h * 0.30)
    red_top = cv2.inRange(hsv, (0, 80, 50), (15, 255, 255))
    red2_top = cv2.inRange(hsv, (165, 80, 50), (180, 255, 255))
    red_top = cv2.bitwise_or(red_top, red2_top)
    top_crop = red_top[:top_h, :]
    red_centers_top = _find_blob_centroids(top_crop, min_area=300, max_area=10000)
    for cx, cy, area in red_centers_top:
        radius = int(np.sqrt(area) / 2)
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius)
        y2 = min(top_h, cy + radius)
        bw = max(x2 - x1, 8)
        bh = max(y2 - y1, 8)
        xc = (x1 + bw / 2) / w
        yc = (y1 + bh / 2) / h
        wn = bw / w
        hn = bh / h
        labels.append((3, xc, yc, wn, hn))  # traffic_light_red

    # Stop signs / speed limits: below traffic light zone
    below_top = red_mask[top_h:, :]
    # Find contours directly from the mask (for shape analysis)
    contours_below, _ = cv2.findContours(below_top, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_below:
        area = cv2.contourArea(cnt)
        if not (300 <= area <= 50000):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cy_full = cy + top_h

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        bw = max(x2 - x1, 10)
        bh = max(y2 - y1, 10)
        xc = (x1 + bw / 2) / w
        yc = (y1 + bh / 2) / h
        wn = bw / w
        hn = bh / h

        # --- Shape classification ---
        # Perimeter & circularity
        peri = cv2.arcLength(cnt, True)
        circularity = 4.0 * math.pi * area / (peri * peri) if peri > 0 else 0.0

        # Approximate contour to polygon
        epsilon = 0.04 * peri       # sensitivity (higher = less detail)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        n_vertices = len(approx)

        # Bounding-box aspect ratio (stop signs are ~square)
        aspect = bw / bh if bh > 0 else 1.0

        # Classification heuristics (tolerant to perspective distortion):
        #   Stop sign: 4+ vertices, circularity 0.5–0.9, near-square 0.7–1.4
        #   Speed limit: circular, circularity > 0.72, area typically smaller
        if (n_vertices >= 4 and 0.50 <= circularity <= 0.90
                and 0.70 <= aspect <= 1.40):
            cls = 0              # stop (octagon / near-square, incl. YIELD)
        elif circularity > 0.72:
            cls = 1              # speed_limit (circular)
        else:
            # Fallback: use area (larger → stop sign, smaller → speed limit)
            cls = 0 if area >= 2500 else 1

        labels.append((cls, xc, yc, wn, hn))

    # ------------------------------------------------------------------
    # 2. YELLOW objects (traffic light yellow)
    # ------------------------------------------------------------------
    yellow_top = cv2.inRange(hsv, (18, 60, 100), (35, 255, 255))
    top_crop_y = yellow_top[:top_h, :]
    y_centers = _find_blob_centroids(top_crop_y, min_area=300, max_area=10000)
    for cx, cy, area in y_centers:
        radius = int(np.sqrt(area) / 2)
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius)
        y2 = min(top_h, cy + radius)
        bw = max(x2 - x1, 8)
        bh = max(y2 - y1, 8)
        xc = (x1 + bw / 2) / w
        yc = (y1 + bh / 2) / h
        wn = bw / w
        hn = bh / h
        labels.append((4, xc, yc, wn, hn))  # traffic_light_yellow

    # Warning signs: yellow triangles below top zone
    yellow_sign = cv2.inRange(hsv, (18, 60, 80), (35, 255, 255))
    below_y = yellow_sign[top_h:, :]
    y_centers_below = _find_blob_centroids(below_y, min_area=500, max_area=30000)
    for cx, cy, area in y_centers_below:
        cy_full = cy + top_h
        radius = int(np.sqrt(area) / 2)
        x1 = max(0, cx - radius)
        y1 = max(0, cy_full - radius)
        x2 = min(w, cx + radius)
        y2 = min(h, cy_full + radius)
        bw = max(x2 - x1, 10)
        bh = max(y2 - y1, 10)
        xc = (x1 + bw / 2) / w
        yc = (y1 + bh / 2) / h
        wn = bw / w
        hn = bh / h
        labels.append((2, xc, yc, wn, hn))  # priority_warning

    # ------------------------------------------------------------------
    # 3. GREEN objects (traffic light green)
    # ------------------------------------------------------------------
    # Webots green is slightly shifted; use two bands for coverage
    green_band1 = cv2.inRange(hsv, (35, 50, 50), (70, 255, 255))
    green_band2 = cv2.inRange(hsv, (75, 50, 50), (90, 255, 200))
    green_mask = cv2.bitwise_or(green_band1, green_band2)
    top_crop_g = green_mask[:top_h, :]
    g_centers = _find_blob_centroids(top_crop_g, min_area=300, max_area=10000)
    for cx, cy, area in g_centers:
        radius = int(np.sqrt(area) / 2)
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius)
        y2 = min(top_h, cy + radius)
        bw = max(x2 - x1, 8)
        bh = max(y2 - y1, 8)
        xc = (x1 + bw / 2) / w
        yc = (y1 + bh / 2) / h
        wn = bw / w
        hn = bh / h
        labels.append((5, xc, yc, wn, hn))  # traffic_light_green

    # ------------------------------------------------------------------
    # 4. BLUE objects → priority road signs (blue circle)
    # ------------------------------------------------------------------
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    below_b = blue_mask[top_h:, :]
    b_centers = _find_blob_centroids(below_b, min_area=500, max_area=20000)
    for cx, cy, area in b_centers:
        cy_full = cy + top_h
        radius = int(np.sqrt(area) / 2)
        x1 = max(0, cx - radius)
        y1 = max(0, cy_full - radius)
        x2 = min(w, cx + radius)
        y2 = min(h, cy_full + radius)
        bw = max(x2 - x1, 10)
        bh = max(y2 - y1, 10)
        xc = (x1 + bw / 2) / w
        yc = (y1 + bh / 2) / h
        wn = bw / w
        hn = bh / h
        labels.append((2, xc, yc, wn, hn))  # priority_warning

    # Deduplicate overlapping boxes (allow different classes to coexist)
    labels = _dedup_labels(labels, iou_thresh=0.6)

    # Remove edge artifacts (ground reflections, cut-off detections)
    labels = _filter_edge_detections(labels, h, w)

    return labels


# ---------------------------------------------------------------------------
# Frame capture & save
# ---------------------------------------------------------------------------

def try_capture(frame, timestamp):
    """
    If dataset mode is on and enough time has elapsed, save one frame.

    Returns:
        (updated_counter, debug_overlay_frame)
        debug_overlay_frame is None if no capture happened,
        or a BGR copy of frame with bounding boxes drawn if a frame was saved.
    """
    global dataset_counter, last_capture_time

    if not dataset_mode:
        return dataset_counter, None

    # Rate-limit: 2 seconds between captures
    if timestamp - last_capture_time < 2.0:
        return dataset_counter, None

    last_capture_time = timestamp

    labels = label_frame(frame)

    # Filter out edge artifacts (ground reflections, cut-off detections)
    h, w = frame.shape[:2]
    labels = _filter_edge_detections(labels, h, w)

    # Save image
    dataset_counter += 1
    fn = f"frame_{dataset_counter:06d}"
    img_path = os.path.join(IMG_DIR, f"{fn}.jpg")
    cv2.imwrite(img_path, frame)

    # Save YOLO labels (empty .txt = hard negative sample — useful for training)
    txt_path = os.path.join(LAB_DIR, f"{fn}.txt")
    with open(txt_path, "w") as f:
        for cls, xc, yc, wn, hn in labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    # Build debug overlay with bounding boxes
    debug_frame = frame.copy()
    if labels:
        for cls, xc, yc, wn, hn in labels:
            x1 = int((xc - wn / 2) * frame.shape[1])
            y1 = int((yc - hn / 2) * frame.shape[0])
            bw = int(wn * frame.shape[1])
            bh = int(hn * frame.shape[0])
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(debug_frame, (x1, y1), (x1 + bw, y1 + bh), color, 2)
            name = CLASS_NAMES.get(cls, f"class{cls}")
            cv2.putText(debug_frame, name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        label_str = f"{len(labels)} obj"
    else:
        label_str = "no obj"

    # Counter + mode indicator
    cv2.putText(debug_frame, f"[DS] #{dataset_counter}  {label_str}",
                (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    print(f"  [DATASET] {fn}.jpg + .txt ({len(labels)} objects)")

    return dataset_counter, debug_frame
