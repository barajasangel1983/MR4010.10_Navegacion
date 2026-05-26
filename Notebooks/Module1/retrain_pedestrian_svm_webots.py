"""
retrain_pedestrian_svm_webots.py
---------------------------------
Retrains the pedestrian HOG-SVM model using Webots-specific training data.

ROOT CAUSE OF POOR DETECTION
-----------------------------
The original model was trained on the DC (Daimler Challenges) pedestrian dataset:
real-world photographs of humans in natural clothing. HOG gradients in those images
come from clothing wrinkles, fabric texture, and lighting variations across the body.

Webots pedestrians are flat-color 3D renders (solid green shirt, blue pants). HOG
gradients in synthetic renders exist mainly at the sharp color-boundary edges, not
spread throughout the body. The SVM learned the wrong gradient distribution, so it
rarely scores Webots pedestrians above the detection threshold.

FIX STRATEGY
-------------
1. Auto-locate pedestrians in Webots screenshots using the distinctive green shirt
   color (HSV green blob detection).
2. Extract positive crops at the sizes the detector actually uses.
3. Extract hard negative crops from the same frames (road, buildings, sky).
4. Mix with DC positive examples to keep general body-shape knowledge.
5. Retrain with LinearSVC (faster inference, better generalization than RBF SVM).
6. Overwrite the existing model file.

HOW TO USE
-----------
1. Collect at least 5-10 Webots screenshots that contain pedestrians.
   In the simulator press 'A' (keyboard) or Square (PS4) to save a frame to src/.
2. Run this script from the project root:
       python Notebooks/Module1/retrain_pedestrian_svm_webots.py
3. The new model is saved to models/pedestrian_svm_hog.pkl.
   No changes to the controller are needed — it loads the model by path at startup.

EXPECTED IMPROVEMENT
---------------------
With as few as 10-20 Webots positive crops the new model should score synthetic
pedestrians well above the 0.3 confidence threshold used in V1.6.
"""

import os
import glob
import pickle
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ==============================
# PATHS
# ==============================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
DC_BASE      = os.path.join(PROJECT_ROOT, "data", "data_svm", "DC-ped-dataset_base")
MODEL_OUT    = os.path.join(PROJECT_ROOT, "models", "pedestrian_svm_hog.pkl")

# ==============================
# HOG CONFIG — must match detect_pedestrians_svm in the controller exactly
# ==============================

MODEL_W = 18   # width fed to HOG (pixels)
MODEL_H = 36   # height fed to HOG (pixels)

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    transform_sqrt=True,
    visualize=False,
    feature_vector=True,
)

# ==============================
# GREEN-SHIRT HSV RANGE
# Webots default pedestrian: bright solid green shirt.
# Adjust if your pedestrian model uses a different color.
# ==============================

GREEN_LOW  = np.array([35, 100, 100])   # HSV lower bound
GREEN_HIGH = np.array([85, 255, 255])   # HSV upper bound

# Minimum green blob area (px2) to accept as a pedestrian candidate
GREEN_BLOB_MIN_AREA = 200


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_hog(gray_crop):
    """Return HOG feature vector for a grayscale crop resized to MODEL_W x MODEL_H."""
    resized = cv2.resize(gray_crop, (MODEL_W, MODEL_H))
    return hog(resized, **HOG_PARAMS)


# ==============================
# AUTO-LABEL WEBOTS SCREENSHOTS
# ==============================

def find_pedestrian_boxes_in_frame(frame_bgr, min_area=GREEN_BLOB_MIN_AREA):
    """
    Use green-shirt color to locate pedestrian bounding boxes in a Webots frame.
    Returns list of (x, y, w, h) boxes around detected pedestrian regions.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)

    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # The green blob is the shirt (torso). Expand to the full pedestrian body:
        #   upward ~h for the head, downward ~2h for the legs, 30% side for arms.
        head_h = int(h * 1.0)
        leg_h  = int(h * 2.0)
        side_w = int(w * 0.4)

        x0 = max(0, x - side_w)
        y0 = max(0, y - head_h)
        x1 = min(frame_bgr.shape[1], x + w + side_w)
        y1 = min(frame_bgr.shape[0], y + h + leg_h)

        if (x1 - x0) > 10 and (y1 - y0) > 10:
            boxes.append((x0, y0, x1 - x0, y1 - y0))

    return boxes


def extract_positive_crops(frame_bgr, boxes):
    """Crop pedestrian regions from a frame and return as grayscale arrays."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    crops = []
    for (x, y, w, h) in boxes:
        crop = gray[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        crops.append(crop)
    return crops


def extract_negative_crops(frame_bgr, ped_boxes, n_samples=20, win_w=30, win_h=60):
    """
    Randomly sample non-pedestrian crops from a frame for hard negatives.
    Rejects any crop that overlaps a known pedestrian box by more than 10%.
    """
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    crops  = []
    tries  = 0

    while len(crops) < n_samples and tries < n_samples * 20:
        tries += 1
        x = np.random.randint(0, max(1, width  - win_w))
        y = np.random.randint(0, max(1, height - win_h))

        overlap = False
        for (px, py, pw, ph) in ped_boxes:
            ix0 = max(x, px);      iy0 = max(y, py)
            ix1 = min(x+win_w, px+pw); iy1 = min(y+win_h, py+ph)
            inter = max(0, ix1-ix0) * max(0, iy1-iy0)
            if inter > 0.10 * win_w * win_h:
                overlap = True
                break

        if not overlap:
            crops.append(gray[y:y+win_h, x:x+win_w])

    return crops


# ==============================
# LOAD DC DATASET
# ==============================

def load_dc_positives(dc_base, max_per_split=200):
    """Load positive (pedestrian) PGM examples from the DC-ped-dataset."""
    features = []
    for split in ["1", "2", "3"]:
        ped_dir = os.path.join(dc_base, split, "ped_examples")
        if not os.path.isdir(ped_dir):
            continue
        files = glob.glob(os.path.join(ped_dir, "*.pgm"))[:max_per_split]
        for path in files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (MODEL_H, MODEL_W):
                img = cv2.resize(img, (MODEL_W, MODEL_H))
            features.append(hog(img, **HOG_PARAMS))
    return features


def load_dc_negatives(dc_base, max_per_split=200):
    """Load negative (non-pedestrian) PGM examples from the DC-ped-dataset."""
    features = []
    for split in ["1", "2", "3"]:
        neg_dir = os.path.join(dc_base, split, "non-ped_examples")
        if not os.path.isdir(neg_dir):
            continue
        files = glob.glob(os.path.join(neg_dir, "*.pgm"))[:max_per_split]
        for path in files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (MODEL_H, MODEL_W):
                img = cv2.resize(img, (MODEL_W, MODEL_H))
            features.append(hog(img, **HOG_PARAMS))
    return features


# ==============================
# MAIN TRAINING PIPELINE
# ==============================

def main():
    print("=" * 60)
    print("Webots pedestrian SVM retraining script")
    print("=" * 60)

    # 1. Find Webots screenshots
    png_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
    if not png_files:
        print(f"[ERROR] No PNG screenshots found in {SRC_DIR}")
        print("        Save frames from Webots (press 'A') and rerun.")
        return

    print(f"\nFound {len(png_files)} Webots screenshots in {SRC_DIR}")

    # 2. Extract positive and negative crops from Webots frames
    webots_pos_features = []
    webots_neg_features = []

    for path in png_files:
        frame = cv2.imread(path)
        if frame is None:
            continue

        boxes     = find_pedestrian_boxes_in_frame(frame)
        pos_crops = extract_positive_crops(frame, boxes) if boxes else []
        neg_crops = extract_negative_crops(frame, boxes, n_samples=15)

        for crop in pos_crops:
            webots_pos_features.append(extract_hog(crop))
        for crop in neg_crops:
            webots_neg_features.append(extract_hog(crop))

        print(f"  {os.path.basename(path)}: "
              f"{len(boxes)} pedestrian(s) -> "
              f"{len(pos_crops)} pos, {len(neg_crops)} neg crops")

    print(f"\nWebots positives: {len(webots_pos_features)}")
    print(f"Webots negatives: {len(webots_neg_features)}")

    if len(webots_pos_features) == 0:
        print("\n[WARNING] No pedestrian crops found via green-shirt color.")
        print("  - Check GREEN_LOW / GREEN_HIGH match your pedestrian model color.")
        print("  - Make sure screenshots contain pedestrians.")
        print("Proceeding with DC dataset only (no domain-specific data).")

    # 3. Load DC dataset
    print("\nLoading DC dataset...")
    dc_pos = load_dc_positives(DC_BASE, max_per_split=300)
    dc_neg = load_dc_negatives(DC_BASE, max_per_split=300)
    print(f"DC positives: {len(dc_pos)}, DC negatives: {len(dc_neg)}")

    # 4. Assemble training set
    # Repeat Webots crops 3x so the SVM prioritises the synthetic domain
    # even when DC samples outnumber them.
    WEBOTS_REPEAT = 3

    X_pos = webots_pos_features * WEBOTS_REPEAT + dc_pos
    X_neg = webots_neg_features * WEBOTS_REPEAT + dc_neg

    X = np.array(X_pos + X_neg, dtype=np.float32)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.float32)

    print(f"\nTotal: {len(X_pos)} positives, {len(X_neg)} negatives")

    if len(X_pos) == 0 or len(X_neg) == 0:
        print("[ERROR] Cannot train — one class is empty.")
        return

    # 5. Train LinearSVC pipeline
    # LinearSVC generalises better than RBF SVM for HOG features because HOG
    # vectors are already in a meaningful metric space. The original RBF model
    # had 6721 support vectors (nearly the full training set), indicating
    # overfitting. LinearSVC also runs much faster at inference time.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)),
    ])

    print("\nRunning 3-fold cross-validation...")
    scores = cross_val_score(model, X, y, cv=3, scoring="f1")
    print(f"  F1 scores: {scores}  ->  mean {scores.mean():.3f}")

    print("Fitting final model on all data...")
    model.fit(X, y)

    # 6. Save model
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved -> {MODEL_OUT}")

    # 7. Quick self-test on the first screenshot
    if png_files:
        frame = cv2.imread(png_files[0])
        boxes = find_pedestrian_boxes_in_frame(frame)
        print(f"\nSelf-test on {os.path.basename(png_files[0])}:")
        if boxes:
            for (x, y, w, h) in boxes:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                feat = extract_hog(gray[y:y+h, x:x+w]).reshape(1, -1)
                score = model.decision_function(feat)[0]
                print(f"  box ({x},{y},{w},{h})  score = {score:.3f}")
        else:
            print("  No pedestrians detected via color in this frame.")

    print("\nDone. Restart the Webots controller to load the new model.")
    print("Controller uses decision_function > 0.3 (confidence_threshold in V1.6).")


if __name__ == "__main__":
    main()
