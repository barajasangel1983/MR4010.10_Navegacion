# AutonomusCar — Next Session Plan

## Immediate (Next 1-2 Sessions)

### 1. V1.7 Testing
- [ ] Angel pulls repo to local PC
- [ ] Test V1.7 controller in Webots:
  - [ ] Verify grayscale lane detection works
  - [ ] Test PS4 + keyboard control
  - [ ] Toggle dataset mode (Triangle / D key)
  - [ ] Verify images saved to `data/webots_dataset/images/`
  - [ ] Verify labels saved to `data/webots_dataset/labels/`
  - [ ] Check bounding box quality on overlay
- [ ] Report issues → iterate on V1.7

### 2. Dataset Generation
- [ ] Run Webots for extended periods in varied conditions:
  - [ ] Normal lighting
  - [ ] Low lighting (dim headlights)
  - [ ] Different track layouts (curves, intersections, T-junctions)
  - [ ] Multiple traffic signs per session
  - [ ] Multiple traffic light states
- [ ] Target: **minimum 500-1000 images per class**
- [ ] Include hard negatives (empty roads, partial occlusions, distant objects)
- [ ] Track dataset statistics (class distribution, average bbox size)

### 3. YOLO Training Setup
- [ ] Select YOLO variant: **Ultralytics YOLOv8** (recommended)
  - Why: pip installable, well-documented, good pre-trained weights, ONNX export
  - Alternative: YOLOv5 (more community examples), RT-DETR (faster)
- [ ] Install in conda `autonomuscar` env: `pip install ultralytics`
- [ ] Create dataset config: `configs/yolov8_dataset.yaml`
- [ ] Set up training script: `scripts/train_yolo.py`
- [ ] Define augmentation pipeline (brightness, contrast, blur, mosaic, copy-paste)

### 4. Training Pipeline
- [ ] Run initial training (50 epochs, pretrained backbone)
- [ ] Monitor mAP, precision, recall per class
- [ ] Analyze confusion matrix
- [ ] Hyperparameter tuning (lr, batch size, epochs)
- [ ] Export ONNX for future Jetson deployment

## Mid-Term (Weeks 2-3)

### 5. Model Integration
- [ ] Create Webots perception node that loads YOLO model
- [ ] Feed detections into V1.7 controller decision logic
- [ ] Implement control responses:
  - Stop sign → brake, halt
  - Speed limit → reduce speed
  - Traffic light → stop/go based on color
  - Priority sign → yield behavior
- [ ] Test end-to-end: perception → decision → control

### 6. Data Quality Improvements
- [ ] Add data augmentation during training
- [ ] Balance dataset (oversample underrepresented classes)
- [ ] Add random crops and rotations to increase variety
- [ ] Consider domain randomization in Webots (vary textures, lighting)

## Long-Term (Future)

### 7. Real-World Deployment
- [ ] Port architecture to Jetson Nano
- [ ] Replace Webots camera with real camera feed
- [ ] Collect real-world data for fine-tuning
- [ ] Add depth estimation (stereo or monocular)
- [ ] Migrate to ROS2 (Humble) when upgrading to Orin Nano

## Decisions to Make
1. YOLO variant selection (YOLOv8 vs YOLOv5 vs others)
2. Target model size (nano/small/medium) — balance speed vs accuracy
3. Training hardware (VPS CPU vs local GPU)
4. Dataset size targets per class
5. Real-world data collection strategy (when to start)
