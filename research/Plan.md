# Project Plan — MR4010.10 Navigation (AutonomusCar)

## Phase 1: Foundation
- [x] VPS setup (AMD EPYC, 7.8GB RAM, Ubuntu 22.04)
- [x] Anaconda installation + `autonomuscar` conda env (Python 3.11)
- [x] Jupyter tunnel workflow established
- [x] SSH key auth: `id_ed25519_autonomuscar` for Jetson
- [x] Git sync workflow between local PC → VPS → Jetson

## Phase 2: Webots Controller (V1.7)
- [x] V1.6 base controller established (grayscale + PID + PS4)
- [x] V1.7 added dataset collection module
  - [x] `dataset_mode.py` — color/shape heuristic labeling
  - [x] YOLO format output (class + normalized bbox)
  - [x] Trigger: PS4 Triangle / keyboard `D`
  - [x] Auto-capture every 2 seconds
  - [x] Hard-negative samples (empty `.txt`)
  - [x] Debug overlay with capture counter
- [ ] Angel tests V1.7 in Webots on PC
- [ ] Iterate on dataset mode (capture rate, labeling accuracy, lighting conditions)

## Phase 3: Dataset Collection
- [ ] Generate Webots dataset with varied conditions:
  - [ ] Multiple track layouts
  - [ ] Different lighting settings
  - [ ] Traffic signs (stop, speed limit, priority)
  - [ ] Traffic lights (red, yellow, green)
  - [ ] Curved roads, intersections, T-junctions
- [ ] Aim for minimum 500-1000 labeled images per class
- [ ] Include hard negatives (no objects, partial views, occluded)
- [ ] Store in `data/webots_dataset/` (images + labels/)

## Phase 4: YOLO Training
- [x] **Plan created:** `research/AutonomusCar/yolo_training_plan.md`
- [ ] Choose YOLO variant (Ultralytics YOLOv8 recommended)
  - Start with **YOLOv8s** (balance of speed/accuracy)
  - Scale to **YOLOv8n** if compute is limiting
- [ ] Prepare dataset config (`configs/webots_dataset.yaml`)
- [ ] Training pipeline:
  - [ ] Data splitting (70/15/15 train/val/test)
  - [ ] Augmentation (brightness, contrast, blur, mosaic, copy-paste)
  - [ ] Transfer learning (pretrained COCO weights)
  - [ ] Hyperparameter tuning (lr, epochs, batch size)
- [ ] Validation: mAP@0.5, precision, recall per class
- [ ] Export formats: ONNX (general), TensorRT (Jetson/Orin)
- [ ] Target metrics: mAP >40%, inference >30 FPS on VPS CPU
- [ ] Full plan: `research/AutonomusCar/yolo_training_plan.md`

## Phase 5: Integration
- [ ] Deploy trained YOLO model as Webots perception node
- [ ] Feed detections into V1.7 controller decision logic
- [ ] Implement control strategy based on detected objects:
  - Stop sign → halt
  - Speed limit → adjust speed
  - Traffic light → stop/go based on color
  - Priority sign → yield behavior

## Phase 6: Real-World Transfer (Future)
- [ ] Port architecture to Jetson Nano
- [ ] Replace Webots camera with real camera feed
- [ ] Fine-tune YOLO with real-world data
- [ ] Add depth estimation (stereo cameras or monocular)
- [ ] ROS2 (Humble) migration when upgrading to Orin Nano
