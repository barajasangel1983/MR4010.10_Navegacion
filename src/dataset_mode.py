# ============================================================
# Dataset Mode Module — Simple Controller V1.7
# Auto-capture camera frames + color/shape heuristic labeling
# for YOLO-based traffic sign / traffic light detector training.
#
# Trigger: PS4 Triangle (btn 2) or Keyboard 'd' / 'D'
# Output: data/webots_dataset/images/*.jpg + labels/*.txt
# ============================================================

import cv2
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
    red_top = cv2.inRange(hsv, (0, 150, 80), (15, 255, 255))
    red2_top = cv2.inRange(hsv, (160, 150, 80), (180, 255, 255))
    red_top = cv2.bitwise_or(red_top, red2_top)
    top_crop = red_top[:top_h, :]
    red_centers_top = _find_blob_centroids(top_crop, min_area=150, max_area=10000)
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
    red_centers_below = _find_blob_centroids(below_top, min_area=300, max_area=50000)
    for cx, cy, area in red_centers_below:
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
        aspect = bw / bh if bh > 0 else 1.0
        # Speed limit = circular, Stop = octagonal (similar in pixels)
        # Heuristic: smaller → speed limit, larger → stop sign
        cls = 1 if area < 5000 else 0
        labels.append((cls, xc, yc, wn, hn))

    # ------------------------------------------------------------------
    # 2. YELLOW objects (traffic light yellow)
    # ------------------------------------------------------------------
    yellow_top = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    top_crop_y = yellow_top[:top_h, :]
    y_centers = _find_blob_centroids(top_crop_y, min_area=150, max_area=10000)
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
    yellow_sign = cv2.inRange(hsv, (20, 80, 120), (35, 255, 255))
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
    green_top = cv2.inRange(hsv, (40, 100, 80), (70, 255, 255))
    top_crop_g = green_top[:top_h, :]
    g_centers = _find_blob_centroids(top_crop_g, min_area=150, max_area=10000)
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

    # Deduplicate overlapping boxes
    labels = _dedup_labels(labels, overlap_iou_thresh=0.6)

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
