"""
============================================================
train_pedestrian_svm.py
Equipo 17 - Actividad 2.2 (Navegacion Autonoma)

Entrena un clasificador SVM lineal con descriptores HOG para
detectar peatones. Dataset: INRIA Person en formato Pascal VOC.

Salidas:
  - pedestrian_svm.pkl       (modelo entrenado para Webots)
  - confusion_matrix.png     (matriz de confusion para el video)

Pipeline:
  1. Lee XMLs y recorta cada persona como POSITIVO (64x128).
  2. De cada imagen toma 10 ventanas aleatorias que NO toquen
     a ninguna persona como NEGATIVOS.
  3. Calcula HOG de todo y entrena un LinearSVC.
  4. Evalua sobre el set de Test y guarda matriz + modelo.
============================================================
"""

import os
import glob
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)


# ============================================================
# CONFIGURACION
# ============================================================

# Ruta absoluta al dataset (cambia esto si lo mueves)
DATASET_DIR = r"D:\Maestria\5to Trimetre\Navegación Autónoma\Semana 5\Dataset"

# Tamano estandar HOG para personas (Dalal & Triggs 2005)
WIN_W, WIN_H = 64, 128

# Cuantas ventanas negativas tomar de cada imagen
NEG_PER_IMAGE = 10

# Parametros HOG clasicos
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True,
)

# Semillas para reproducibilidad
random.seed(42)
np.random.seed(42)


# ============================================================
# CARGA DE IMAGENES Y BOUNDING BOXES
# ============================================================

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    """
    Reemplaza a cv2.imread() para soportar rutas con acentos en Windows.
    OpenCV en Windows usa ANSI y se rompe con caracteres Unicode (ej. 'Navegacion').
    Aqui leemos los bytes con Python (que si maneja Unicode) y luego los decodificamos.
    """
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(stream, flags)
    except Exception:
        return None


def parse_xml_bboxes(xml_path):
    """Devuelve lista de (xmin, ymin, xmax, ymax) para cada persona."""
    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.findall("object"):
        if obj.find("name").text == "person":
            b = obj.find("bndbox")
            boxes.append((
                int(b.find("xmin").text),
                int(b.find("ymin").text),
                int(b.find("xmax").text),
                int(b.find("ymax").text),
            ))
    return boxes


def load_positives(images_dir, annotations_dir):
    """Recorta cada persona anotada y la redimensiona a 64x128."""
    crops = []
    for xml_file in glob.glob(os.path.join(annotations_dir, "*.xml")):
        name = os.path.splitext(os.path.basename(xml_file))[0]
        img_path = os.path.join(images_dir, name + ".png")
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape
        for (x1, y1, x2, y2) in parse_xml_bboxes(xml_file):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 40:
                continue
            person = img[y1:y2, x1:x2]
            person = cv2.resize(person, (WIN_W, WIN_H))
            crops.append(person)
    return crops


def box_overlaps_any(box, others):
    """Regresa True si box (x1,y1,x2,y2) intersecta a alguna otra caja."""
    x1, y1, x2, y2 = box
    for (ox1, oy1, ox2, oy2) in others:
        if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
            return True
    return False


def load_negatives(images_dir, annotations_dir, n_per_image=NEG_PER_IMAGE):
    """Saca n ventanas 64x128 aleatorias por imagen, fuera de las personas."""
    crops = []
    for xml_file in glob.glob(os.path.join(annotations_dir, "*.xml")):
        name = os.path.splitext(os.path.basename(xml_file))[0]
        img_path = os.path.join(images_dir, name + ".png")
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape
        if w < WIN_W or h < WIN_H:
            continue
        person_boxes = parse_xml_bboxes(xml_file)
        added = 0
        attempts = 0
        while added < n_per_image and attempts < n_per_image * 5:
            attempts += 1
            x = random.randint(0, w - WIN_W)
            y = random.randint(0, h - WIN_H)
            cand = (x, y, x + WIN_W, y + WIN_H)
            if box_overlaps_any(cand, person_boxes):
                continue
            crops.append(img[y:y + WIN_H, x:x + WIN_W])
            added += 1
    return crops


# ============================================================
# FEATURES HOG
# ============================================================

def extract_hog_features(images):
    """Aplica HOG a cada imagen y regresa un array (N, n_features)."""
    return np.array([hog(im, **HOG_PARAMS) for im in images])


# ============================================================
# MAIN
# ============================================================

def main():
    train_imgs = os.path.join(DATASET_DIR, "Train", "JPEGImages")
    train_ann = os.path.join(DATASET_DIR, "Train", "Annotations")
    test_imgs = os.path.join(DATASET_DIR, "Test", "JPEGImages")
    test_ann = os.path.join(DATASET_DIR, "Test", "Annotations")

    print("== Cargando dataset INRIA ==")

    print("Train positivos...")
    pos_train = load_positives(train_imgs, train_ann)
    print(f"  {len(pos_train)} recortes")

    print("Train negativos...")
    neg_train = load_negatives(train_imgs, train_ann)
    print(f"  {len(neg_train)} recortes")

    print("Test positivos...")
    pos_test = load_positives(test_imgs, test_ann)
    print(f"  {len(pos_test)} recortes")

    print("Test negativos...")
    neg_test = load_negatives(test_imgs, test_ann)
    print(f"  {len(neg_test)} recortes")

    print("\n== Calculando HOG ==")
    X_train = np.vstack([
        extract_hog_features(pos_train),
        extract_hog_features(neg_train),
    ])
    y_train = np.hstack([
        np.ones(len(pos_train)),
        np.zeros(len(neg_train)),
    ])
    X_test = np.vstack([
        extract_hog_features(pos_test),
        extract_hog_features(neg_test),
    ])
    y_test = np.hstack([
        np.ones(len(pos_test)),
        np.zeros(len(neg_test)),
    ])
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test : {X_test.shape}")

    print("\n== Entrenando LinearSVC ==")
    clf = LinearSVC(C=1.0, max_iter=5000)
    clf.fit(X_train, y_train)

    print("\n========== RESULTADOS ==========")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusion:")
    print(cm)
    print()
    print(classification_report(y_test, y_pred, target_names=["no-peaton", "peaton"]))

    # ---- Guardar matriz de confusion como PNG ----
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no-peaton", "peaton"])
    ax.set_yticklabels(["no-peaton", "peaton"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion - SVM peaton (HOG)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    print("Matriz guardada: confusion_matrix.png")

    # ---- Guardar el modelo ----
    joblib.dump(clf, "pedestrian_svm.pkl")
    print("Modelo guardado: pedestrian_svm.pkl")


if __name__ == "__main__":
    main()
