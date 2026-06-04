# State of the Nation — MR4010.10 Navigation

## Current Status
Work is focused on building a Webots-based autonomous vehicle simulation pipeline with YOLO detection training. The infrastructure (VPS, Anaconda, Git sync) is established. The V1.7 controller with dataset collection is ready for testing.

## Milestones Hit
1. VPS infrastructure set up and accessible via Jupyter tunnel
2. SSH key auth configured for Jetson Nano (key separation maintained)
3. Git sync workflow operational between local PC → VPS → Jetson
4. V1.6 controller (grayscale + PID) verified working in Webots
5. V1.7 controller with dataset collection module pushed to repo
6. Dataset module supports color/shape heuristics for traffic signs + lights
7. YOLO format output with auto-labeling via Webots projection
8. Hard-negative samples included for robust training

## In Progress
- V1.7 testing by Angel in Webots (PC)
- Dataset generation with varied conditions
- **YOLO training plan complete** → `research/AutonomusCar/yolo_training_plan.md`

## Blocked
- None at this time

## Known Issues
- Webots lighting conditions may affect color-based labeling accuracy
- Need to validate bounding box projection math against real camera parameters
- Glibc 2.27 on Jetson limits tooling choices (Node 16 max, ROS Melodic)

## Next Steps
1. Angel pulls and tests V1.7 in Webots
2. Adjust dataset capture settings based on test results
3. Generate initial dataset (~500-1000 images per class)
4. YOLO setup on VPS: `pip install ultralytics`, create dataset YAML, training script
5. Train YOLOv8s from COCO weights, target mAP@0.5 >40%
6. Validate → tune → export ONNX → integrate into Webots controller

> **YOLO Plan:** Full pipeline documented in `research/AutonomusCar/yolo_training_plan.md`
