# ROS Melodic — Current State

## Last Updated
2025-06-04

## Current Status
ROS Melodic is established on the Jetson Nano as the robotics middleware layer. No active ROS development in this session — focus is on Webots simulation and YOLO training pipeline.

## Jetson Constraints
- Ubuntu 18.04 (glibc 2.27)
- ROS Melodic (default for 18.04)
- Node.js 16 max (glibc limitation)
- Python 3.6.9 system (Anaconda provides newer)
- Full OpenClaw node / ROS Humble require Orin Nano upgrade

## ROS Package Structure (Planned)
```
catkin_ws/src/
├── autonomuscar_bringup/          ← Launch files, config
├── autonomuscar_camera/           ← Camera driver + image publishing
├── autonomuscar_perception/       ← YOLO detection node (future)
├── autonomuscar_control/          ← Motor/sensor control nodes
├── autonomuscar_navigation/       ← Path planning, PID controller
└── autonomuscar_msgs/             ← Custom message definitions
```

## Notes
- ROS work deferred until simulation pipeline is stable
- Architecture planning maintained in `Architechture_northstar.md`
- Integration with YOLO detection will create `autonomuscar_perception` package
