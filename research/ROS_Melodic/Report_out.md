# ROS Melodic — ROS-Specific Summary

## Current Focus
ROS Melodic is established on Jetson Nano but not actively developed this session. Focus is Webots simulation → YOLO training → model integration.

## Key Constraints
- Ubuntu 18.04, glibc 2.27
- ROS Melodic (default, limited)
- Orin Nano upgrade needed for ROS Humble + full OpenClaw

## Future ROS Work
- Perception pipeline: YOLO detection node → control decisions
- Custom messages for traffic signs, lights, detections
- Camera driver and motor control nodes
- Full pipeline integration with RViz visualization

## Integration Points
- YOLO model will become `autonomuscar_perception` ROS package
- Webots controller logic will be ported to ROS control nodes
- Dataset pipeline (Webots → YOLO) is independent of ROS layer
