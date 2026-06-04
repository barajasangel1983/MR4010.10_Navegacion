# AutonomusCar — Architecture Northstar

## Vision
A modular autonomous vehicle system where:
- **Simulation** (Webots) enables rapid iteration on perception + control
- **YOLO detection** provides real-time object recognition
- **PID + rule-based control** handles navigation and decision-making
- **Jetson Nano** runs the real-time stack (motor control, perception)
- **DGX/OpenClaw** provides high-level reasoning and orchestration

## System Layers

### Layer 1: Hardware (Jetson Nano → Orin Nano)
- NVIDIA Jetson Nano (current) → Orin Nano (upgrade target)
- Camera: USB/CSI camera (Webots simulated now)
- Motors: DC motors with H-bridge driver
- Sensors: Lidar (future), IMU (future), encoders (future)
- Compute: 128-core Maxwell GPU, 4GB RAM (Nano) → 2048-core GPU, 16GB RAM (Orin)

### Layer 2: Middleware (ROS Melodic → Humble)
- ROS 1 Melodic (current, 18.04) → ROS 2 Humble (future, 22.04)
- Topics: camera/image, detections, motor_commands, steering_angle
- Services: model_load, calibration, status
- TF tree: camera_link → base_link → odom → map

### Layer 3: Perception (YOLO + Heuristics)
- **Object Detection:** YOLOv8 (Ultralytics) — traffic signs + traffic lights
- **Lane Detection:** Grayscale brightness threshold + Hough lines
- **Fusion:** YOLO detections + lane position → unified scene understanding
- **Output:** class_id, bbox, confidence, timestamp

### Layer 4: Decision & Control
- **Rule Engine:** If/then logic for detected objects
  - Stop sign → brake to 0
  - Speed limit → adjust max speed
  - Red light → stop, wait for green
  - Yellow light → prepare to stop
- **Lane Following:** PID controller on lane deviation
- **Path Planning:** A* / Dijkstra for navigation (future)
- **Fallback:** Line-following when no objects detected

### Layer 5: Orchestration (DGX + OpenClaw)
- **DGX:** High-performance compute for model training, large-scale simulations
- **OpenClaw:** Agent orchestration, task scheduling, monitoring
- **VPS:** Jupyter notebooks, data analysis, CI/CD
- **Data Flow:** Webots → VPS (dataset) → DGX (training) → Jetson (deployment)

## Data Pipeline

```
Webots Simulation
    ↓ (auto-capture + auto-label)
data/webots_dataset/ (images + YOLO labels)
    ↓ (upload/clone)
VPS (anaconda env)
    ↓ (YOLO training)
models/ (trained .pt files)
    ↓ (export ONNX)
models/exported/ (ONNX for Jetson)
    ↓ (rsync/git deploy)
Jetson Nano (real camera + YOLO inference)
```

## Controller Architecture (V1.7+)

```
┌─────────────────────────────────────────────────────────────┐
│                      V1.7 Controller                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐      │
│  │ Camera   │───→│ Grayscale │───→│ Lane Detection   │      │
│  │ Feed     │    │ + Threshold│    │ (Hough Lines)    │      │
│  └──────────┘    └───────────┘    └────────┬─────────┘      │
│                                            │                 │
│  ┌──────────┐    ┌───────────┐             │                 │
│  │ Camera   │───→│ Dataset   │             │                 │
│  │ Feed     │    │ Mode      │             │                 │
│  └──────────┘    └───────────┘             │                 │
│                                            ▼                 │
│  ┌──────────────────────────────────────────────────┐        │
│  │              Lane Deviation (yaw)                  │        │
│  └────────────────────────┬─────────────────────────┘        │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────┐        │
│  │              PID Controller                        │        │
│  │  (Kp, Ki, Kd tuned for track)                      │        │
│  └────────────────────────┬─────────────────────────┘        │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────┐        │
│  │              Steering Output                       │        │
│  └────────────────────────┬─────────────────────────┘        │
│                           │                                   │
│  ┌──────────┐    ┌────────▼─────────┐    ┌──────────────┐    │
│  │ PS4 Ctrl │───→│ Decision Layer   │←───│ YOLO Detections│   │
│  │ (Manual) │    │ (Future: V2.0)   │    │ (Perception) │    │
│  └──────────┘    └──────────────────┘    └──────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Evolution Path

### V1.x (Current — Simulation)
- Webots simulation only
- Grayscale + PID lane following
- Manual control (PS4, keyboard)
- Dataset collection with auto-labeling
- Rule-based decisions (future V2.0)

### V2.0 (Perception Integration)
- YOLO detection in Webots
- Rule engine for traffic signs/lights
- Hybrid control: lane + object awareness
- Testing in varied Webots environments

### V3.0 (Real Hardware)
- Deploy to Jetson Nano
- Real camera feed
- Fine-tune YOLO on real data
- Motor control nodes (ROS)
- PID tuning for real vehicle dynamics

### V4.0 (Orin Upgrade)
- Orin Nano hardware upgrade
- ROS Humble migration
- OpenClaw agent integration
- Multi-agent coordination
- Real-world deployment

## Best Practices

1. **Simulation-first:** Test everything in Webots before touching hardware
2. **Modular design:** Each layer (perception, control, decision) is independent
3. **Data-driven:** Use collected data to improve models, not just manual tuning
4. **Version everything:** Controllers, configs, models tracked in Git
5. **Measure everything:** Log detections, errors, performance metrics
6. **Safety first:** Fallback to manual/line-following when perception fails
7. **Document decisions:** Architecture choices logged in research/ directory

## Open Architecture Questions
- Should we use a neural network for lane following too (end-to-end)?
- When to add sensor fusion (camera + Lidar)?
- How to handle unseen traffic signs (OOD detection)?
- Real-world data collection strategy and volume needed?
- ROS 1 → ROS 2 migration timeline?
