# AutonomusCar — Vehicle-Specific Summary

## Current Focus
Building a simulation-based autonomous driving pipeline: Webots controller (V1.7) → dataset collection → YOLO training → model integration.

## Key Components
- **Controller:** `simple_controller_act_2_1_V1.7.py` — grayscale lane detection, PID steering, PS4 control, autonomous mode
- **Dataset Module:** `dataset_mode.py` — auto-capture with color/shape heuristics, YOLO format labeling
- **Target Hardware:** Jetson Nano → Orin Nano upgrade path
- **Simulation:** Webots (current) → real camera (future)

## Dataset Strategy
- Use Webots physics engine for automatic labeling (no manual annotation)
- Color + shape heuristics for 6 object classes (signs + traffic lights)
- Hard-negative samples for robust training
- Initial target: 500-1000 images per class

## YOLO Training Roadmap
Full plan: `research/AutonomusCar/yolo_training_plan.md`

1. **V1.7 Testing** → Angel tests in Webots, validates dataset collection
2. **Data Generation** → Collect varied conditions (500-1000/class), track stats
3. **YOLO Setup** → `ultralytics` on VPS, `yolov8s` model, dataset YAML config
4. **Training** → COCO-pretrained, 50 epochs, target mAP@0.5 >40%
5. **Integration** → Webots perception node → controller decision logic
6. **Real-World** → Jetson deployment, real camera, fine-tuning (future)

### Quick Reference (YOLO Plan)
- **Model:** YOLOv8s → YOLOv8n if needed
- **Classes:** 6 (stop_sign, speed_limit, priority, traffic_light_r/y/g)
- **Training env:** VPS `anaconda3/envs/autonomuscar`
- **Config file:** `configs/webots_dataset.yaml`
- **Training script:** `scripts/train_yolo.py`
- **Export:** ONNX (Webots) → TensorRT (Jetson/Orin)

## Metrics to Track
- Dataset: images per class, class distribution, bbox size stats
- Training: mAP@0.5, precision, recall per class, confusion matrix
- Controller: detection latency, control smoothness, decision accuracy

## Open Questions
- Best YOLO variant for this use case?
- Target model size (nano/small/medium)?
- How to handle domain gap between Webots and real world?
- When to start real-world data collection?
