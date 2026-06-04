# YOLO Training Pipeline — Complete Plan

## Phase 1: Dataset Preparation

### 1.1 Data Collection (Webots)
- **Goal:** 500-1000 images per class with varied conditions
- **Conditions to vary:**
  - Lighting (day, night, overcast)
  - Camera angles (slight left/right tilt)
  - Distances (near, mid, far)
  - Partial occlusions
  - Curved roads (signs at angles)
  - Multiple objects per frame
- **Hard negatives:** ~500 empty frames (no objects, just road/track)
- **Output structure:**
  ```
  data/webots_dataset/
  ├── images/
  │   ├── frame_000001.jpg
  │   ├── frame_000002.jpg
  │   └── ...
  └── labels/
      ├── frame_000001.txt
      ├── frame_000002.txt
      └── ...
  ```

### 1.2 Dataset Split
- **Training:** 70% (350-700 per class)
- **Validation:** 15% (75-150 per class)
- **Test:** 15% (75-150 per class)
- **Split method:** Random by frame ID (ensure same scene doesn't split across sets)

### 1.3 Data Augmentation (Training Only)
- Brightness/contrast variation (±20%)
- Gaussian noise
- Motion blur (simulate camera shake)
- Horizontal flip (optional — may not apply to traffic signs)
- Random crop + resize
- Mosaic augmentation (YOLO native)
- Copy-paste augmentation (YOLO native)

## Phase 2: Model Selection

### 2.1 Recommended: Ultralytics YOLOv8
**Why:**
- `pip install ultralytics` — zero build steps
- Pre-trained COCO weights available
- Auto-labeling support (if we need to re-label later)
- ONNX export for Jetson
- Excellent documentation + community
- Supports YOLOv8n/s/m/l/x (nano to extra-large)

**Alternative options:**
- YOLOv5 — more community examples, slightly less modern
- RT-DETR — faster inference, harder to train
- YOLOv9/v10 — experimental, less mature

### 2.2 Model Size Selection
| Size | Params | Speed (VPS CPU) | mAP (expected) | Use Case |
|------|--------|-----------------|----------------|----------|
| YOLOv8n | 3.2M | ~50 FPS | 37.3 | Fast iteration, low compute |
| YOLOv8s | 11.2M | ~30 FPS | 44.9 | **Recommended** — balance |
| YOLOv8m | 25.9M | ~18 FPS | 50.2 | If accuracy matters most |
| YOLOv8l | 43.7M | ~12 FPS | 52.9 | Overkill for this project |

**Recommendation:** Start with YOLOv8s, scale down to nano if needed.

## Phase 3: Training Setup

### 3.1 Environment
```bash
# On VPS (anaconda autonomuscar env)
conda activate autonomuscar
pip install ultralytics opencv-python torch torchvision
```

### 3.2 Dataset Configuration (`configs/webots_dataset.yaml`)
```yaml
path: ../data/webots_dataset
train: images/train
val: images/val
test: images/test

# Classes
nc: 6
names:
  0: stop_sign
  1: speed_limit
  2: priority_warning
  3: traffic_light_red
  4: traffic_light_yellow
  5: traffic_light_green
```

### 3.3 Training Script (`scripts/train_yolo.py`)
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8s.pt')  # or yolov8n.pt, yolov8m.pt

# Train
results = model.train(
    data='configs/webots_dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    augment=True,
    name='webots_traffic_detection'
)
```

### 3.4 Initial Training Configuration
- **Epochs:** 50 (patience 10 = early stopping if no improvement)
- **Image size:** 640×640 (default)
- **Batch size:** 16 (adjust based on VPS memory — 7.8GB total, ~4GB available)
- **Optimizer:** SGD (default, good for object detection)
- **Learning rate:** Auto-scaled (YOLO handles this)

## Phase 4: Training & Validation

### 4.1 Metrics to Track
- **mAP@0.5:** Mean Average Precision at IoU 0.5 (primary metric)
- **mAP@0.5:0.95:** Stricter metric
- **Precision/Recall:** Per class
- **Confusion matrix:** Understand misclassifications
- **Loss curves:** Train/val box, cls, dfl loss

### 4.2 Validation Steps
1. Check mAP per class — identify weak classes
2. Analyze confusion matrix — what's being misclassified?
3. Review validation images — are there patterns in failures?
4. Adjust augmentation if overfitting/underfitting

### 4.3 Hyperparameter Tuning (If Needed)
- Learning rate: 0.01 → 0.001 (try lower if unstable)
- Batch size: 8, 16, 32 (maximize without OOM)
- Epochs: 50 → 100 (if still improving)
- Mosaic probability: 1.0 → 0.5 (if too noisy)

## Phase 5: Model Export & Testing

### 5.1 Export Formats
```python
# ONNX (for Jetson/production)
model.export(format='onnx')

# TorchScript (PyTorch runtime)
model.export(format='torchscript')

# TensorRT (Jetson optimized — requires Orin or dedicated GPU)
model.export(format='engine')  # Jetson only
```

### 5.2 Inference Testing
```python
from ultralytics import YOLO

model = YOLO('runs/detect/webots_traffic_detection/weights/best.pt')

# Test on validation set
results = model.val(split='val')

# Test on single image
results = model('test_image.jpg')

# Export ONNX for deployment
model.export(format='onnx')
```

### 5.3 Performance Targets
- **Inference speed:** >30 FPS on VPS CPU
- **mAP@0.5:** >40% (acceptable), >50% (good)
- **Per-class recall:** >70% for all classes

## Phase 6: Integration with Controller

### 6.1 Webots Perception Node
- Load YOLO model (ONNX or PyTorch)
- Subscribe to camera topic
- Run inference on each frame
- Publish detections to control node

### 6.2 Decision Logic (Future — V2.0)
```python
def process_detections(detections):
    # Filter by confidence threshold
    detections = [d for d in detections if d.confidence > 0.5]
    
    # Rule-based responses
    for det in detections:
        if det.class == 'stop_sign':
            apply_brake(1.0)
        elif det.class == 'traffic_light_red':
            apply_brake(1.0)
        elif det.class == 'traffic_light_green':
            release_brake()
        elif det.class == 'speed_limit':
            set_max_speed(det.confidence)
```

### 6.3 Jetson Deployment (Future)
- Export model to ONNX
- Optimize with TensorRT (if Orin Nano)
- Deploy to `autonomuscar_perception` ROS package
- Test real-time performance on hardware

## Timeline & Milestones

| Week | Task | Output |
|------|------|--------|
| 1 | V1.7 testing + dataset generation | 500-1000 images per class |
| 2 | YOLO setup + initial training | First trained model |
| 3 | Validation + hyperparameter tuning | Optimized model (mAP >40%) |
| 4 | Model export + Webots integration | Perception node in Webots |
| 5 | Controller integration | End-to-end detection → control |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Small dataset | Data augmentation, transfer learning from COCO |
| Domain gap (Webots → real) | Collect real-world data later, fine-tune |
| Poor labeling accuracy | Manual review of subset, adjust heuristics |
| Training instability | Lower learning rate, gradient clipping |
| Jetson deployment issues | ONNX format, TensorRT only on Orin |

## Notes

- Start simple: YOLOv8s, 50 epochs, default augmentations
- Iterate based on results, don't over-engineer upfront
- Track everything: model weights, configs, metrics
- Document failures as well as successes
- Keep dataset versioned (Git LFS or DVC if large)
