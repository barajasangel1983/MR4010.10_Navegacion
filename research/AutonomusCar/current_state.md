# AutonomusCar — Current State

## Last Updated
2025-06-04

## Where We Left Off
V1.7 controller with dataset collection module is complete and pushed to the repo. Angel is testing it on his PC in Webots. Next milestone: generate a labeled dataset from Webots, then train a YOLOv8 detection model for traffic signs/lights. Full YOLO training plan created at `research/AutonomusCar/yolo_training_plan.md`.

## Repository Structure
```
MR4010.10_Navegacion/
├── src/
│   ├── simple_controller_act_2_1_V1.7.py   ← Main controller
│   └── dataset_mode.py                      ← Dataset collection module
├── data/
│   └── webots_dataset/                      ← Training data (generated)
│       ├── images/
│       └── labels/
├── models/                                   ← Trained models
├── research/                                 ← Architecture + planning docs
├── configs/                                  ← YOLO + other configs
├── notebooks/                                ← Jupyter analysis
└── scripts/                                  ← Utilities
```

## Controller Features (V1.7)
- **Grayscale brightness threshold** for lane detection
- **PID control** for smooth steering
- **PS4 Bluetooth** manual control (L2/R2 for speed, sticks for steering)
- **Autonomous mode** (Toggle: PS4 X / keyboard `A`)
- **Manual mode** with keyboard (WASD) + PS4
- **Dataset collection mode** (Toggle: PS4 Triangle / keyboard `D`)
  - Captures every 2 seconds
  - Color/shape heuristics for labeling
  - YOLO format output
  - Hard-negative samples
  - Debug overlay

## Dataset Collection Details
- **Classes:**
  - 0: stop_sign (red octagon)
  - 1: speed_limit (red circle with text)
  - 2: priority_warning (yellow diamond)
  - 3: traffic_light_red (red vertical rectangle)
  - 4: traffic_light_yellow (yellow vertical rectangle)
  - 5: traffic_light_green (green vertical rectangle)
- **Format:** `class_id x_center y_center width height` (normalized 0-1)
- **Location:** `data/webots_dataset/images/` + `data/webots_dataset/labels/`
- **Filenames:** `frame_XXXXXX.jpg` / `frame_XXXXXX.txt`

## Known Issues
- Color heuristics may misclassify in different lighting conditions
- Bounding box projection from Webots 3D coordinates needs validation
- No data augmentation during collection yet (will add in training phase)
- Dataset will be small initially (~500-1000 per class) — enough for initial training

## YOLO Training Plan (v1.0 — 2025-06-04)
Full plan: `research/AutonomusCar/yolo_training_plan.md`

### Quick Summary
- **Model:** Ultralytics YOLOv8s (start), scale to YOLOv8n if needed
- **Environment:** VPS `anaconda3/envs/autonomuscar` (AMD EPYC, 7.8GB RAM)
- **Training:** `pip install ultralytics`, train from COCO-pretrained weights
- **Config:** `configs/webots_dataset.yaml` (6 classes, dataset paths)
- **Script:** `scripts/train_yolo.py` (single command launch)
- **Target:** mAP@0.5 >40%, >30 FPS on VPS CPU
- **Export:** ONNX → PyTorch deployment in Webots → TensorRT for Orin Nano

### 6 Classes
| ID | Class | Visual |
|----|-------|--------|
| 0 | stop_sign | Red octagon |
| 1 | speed_limit | Red circle |
| 2 | priority_warning | Yellow diamond |
| 3 | traffic_light_red | Red vertical rect |
| 4 | traffic_light_yellow | Yellow vertical rect |
| 5 | traffic_light_green | Green vertical rect |

### Timeline (Estimated)
| Week | Task |
|------|------|
| 1 | V1.7 testing + dataset collection (500-1000/class) |
| 2 | YOLO setup + first training run |
| 3 | Validation + hyperparameter tuning (mAP >40%) |
| 4 | Model export + Webots perception node integration |
| 5 | Controller integration (detection → decision logic) |

## Hardware Context
- **Simulation:** Webots (current)
- **Target Hardware:** Jetson Nano (Ubuntu 18.04, glibc 2.27, ROS Melodic)
- **Compute Upgrade Path:** Orin Nano (Ubuntu 22.04, ROS Humble, full OpenClaw)
- **VPS:** AMD EPYC 9354P, 7.8GB RAM, 61GB free — suitable for YOLO training
