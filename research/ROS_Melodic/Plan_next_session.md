# ROS Melodic — Next Session Plan

## Current State
ROS Melodic on Jetson Nano is operational but not actively developed this session. Focus is on Webots/YOLO pipeline.

## When ROS Development Resumes

### 1. Package Structure Setup
- [ ] Create `catkin_ws/src/` with standard package layout
- [ ] Define custom messages in `autonomuscar_msgs/`
  - TrafficLightState.msg
  - DetectionResult.msg
  - SignClassification.msg

### 2. Perception Pipeline
- [ ] `autonomuscar_camera/` → camera driver, image topic publishing
- [ ] `autonomuscar_perception/` → load YOLO model, publish detections
- [ ] `autonomuscar_control/` → motor/sensor interface, PID controller

### 3. Integration
- [ ] `autonomuscar_bringup/` → launch all nodes, tf tree
- [ ] Test full pipeline: camera → perception → control
- [ ] Add rviz visualization

### 4. Migration Planning (Long-Term)
- [ ] Orin Nano upgrade (Ubuntu 22.04)
- [ ] ROS Humble migration
- [ ] OpenClaw node integration
- [ ] Multi-agent orchestration (DGX + OpenClaw)

## Notes
- ROS work deferred until simulation pipeline stable
- YOLO model will be ported to Jetson as `autonomuscar_perception` package
