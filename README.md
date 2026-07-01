# MR4010.10 — Navegacion

Webots self-driving car controller developed for the MR4010.10 course. The vehicle navigates autonomously using a layered control stack: PID lane following, Behavioral Cloning (BC), and Conditional Imitation Learning (CIL) for intersection maneuvers — with LiDAR-based obstacle detection and a sensor-driven vehicle evasion state machine.

## Project Structure

```
MR4010.10_Navegacion/
├── src/                          # Controller source files
│   ├── simple_controller_act_2_1_V3.0.py   # Latest controller (production)
│   ├── simple_controller_act_2_1_V2.0.py   # CIL NB30 (torch.jit.trace)
│   ├── simple_controller_act_2_1_V1.xx.py  # Prior versions (V1.0 – V1.16)
│   ├── dataset_mode_v5.py        # Dataset capture module (current)
│   └── dataset_mode_v*.py        # Prior capture versions
├── Notebooks/Module1/            # Jupyter notebooks (ordered by topic)
│   ├── 1–9     Image capture, preprocessing, edge detection, ROI, Hough, PID, PS4
│   ├── 12–15   Linear regression, SVM (vehicles, pedestrians)
│   ├── 16–23   Neural networks with Keras (MLP, CNN)
│   ├── 25–26   CNN on Webots dataset, Behavioral Cloning (DAVE-2)
│   ├── 27      BehaviorCloning — BehaviorClass (5-class behavior model)
│   ├── 28      BehavioralCloning_CIL (CIL model, torch.jit.script export)
│   ├── 29      BC_Evasion_v5 (evasion model training)
│   ├── 30      CIL_Training (NB30, torch.jit.trace export)
│   └── 31      CIL_Training_V2
├── models/                       # Saved model files (large binaries git-ignored)
│   ├── cil_model.pt              # CIL TorchScript (NB28, used by V3.0)
│   ├── bc_evasion_model.pt       # BC evasion TorchScript (NB29)
│   ├── bc_behavior_config.json   # BC behavior model config
│   ├── pedestrian_svm.pkl / pedestrian_svm_hog.pkl
│   └── vehicle_svm_hog.pkl
├── data/                         # Datasets (large image folders git-ignored)
│   ├── behavioral_dataset_<date>/     # BC/CIL capture sessions
│   └── webots_dataset*/               # SVM training data
├── research/                     # Literature and reference material
├── tests/
└── requirements.txt
```

## Controller Versions

| Version | Key feature |
|---------|-------------|
| V1.0–V1.9 | PID lane following, PS4 input, LiDAR, SVM detection |
| V1.10 | 4 distance sensors (FL, FR, RL, RR); sensor-based returning |
| V1.11–V1.14 | BC behavior model (5-class), CIL intersection model, A/B/C/D arbitration |
| **V1.15** | Simplified stack — bc_evasion_model only, 3-state CIL machine (IDLE→ARMED→EXECUTING) |
| V1.16 | Runtime toggles: PID (L key), BC (B key), CIL (I key) |
| V2.0 | CIL switched to NB30 (torch.jit.trace, LongTensor nav_command) |
| **V3.0** | CIL reverted to NB28 (torch.jit.script, int nav_command); CIL LANE_FOLLOW always-on mode |

**Use V3.0** (`simple_controller_act_2_1_V3.0.py`) for all new runs.

## Control Priority Stack (V3.0, highest → lowest)

1. Phase 1 EVADING — vehicle < 7 m: `bc_evasion_model` steers right
2. Phase 2 RETURNING — vehicle cleared: DS_MR sensor guides return to lane
3. LiDAR emergency stop — non-vehicle < 7 m: full brake, hold angle
4. LiDAR slow zone — any obstacle 7–25 m: 50 % speed
5. CIL EXECUTING — nav_cmd ≠ LANE_FOLLOW, line gone, model loaded
6. PID — line detected, no CIL, no vehicle evasion
7. bc_evasion_model Phase 0 — line lost, no CIL, no vehicle

## Keyboard Controls

| Key | Action |
|-----|--------|
| Arrows | Manual steering and speed |
| S | Toggle autonomous mode |
| A | Toggle ADAS Monitor window |
| N | Toggle lane side (LEFT ↔ RIGHT) |
| P | Toggle debug panel |
| D | Toggle dataset capture |
| L | Toggle PID on/off |
| B | Toggle BC evasion on/off |
| I | Toggle CIL on/off |
| T | Toggle CIL LANE_FOLLOW mode |
| C | Save single camera frame |
| Q | Quit |

**PS4:** Left stick = steer, R2 = throttle, L2 = brake, X = autonomous, L1 = LEFT turn, R1 = RIGHT turn, Circle = STRAIGHT, Triangle = CIL LANE_FOLLOW.

## Models

| File | Description |
|------|-------------|
| `cil_model.pt` | CIL TorchScript (NB28, `forward(x, command: int)`) |
| `bc_evasion_model.pt` | BC evasion TorchScript — Phase 0 (line lost) and Phase 1 (EVADING) |
| `pedestrian_svm_hog.pkl` | HOG + SVM pedestrian detector |
| `vehicle_svm_hog.pkl` | HOG + SVM vehicle detector |

Train via the numbered notebooks in `Notebooks/Module1/`. Export to `models/` before running the controller.

## Dataset Capture

Toggle with **D** (keyboard) or **Triangle** (PS4). Captures at 4 Hz during intersection maneuvers, 2 Hz otherwise. Output written to `data/behavioral_dataset_<date>/`:

```
images/   <contributor_id>_frame_XXXXXX.jpg
measurements.csv
```

CSV columns: `contributor_id`, `session_id`, `timestamp`, `image_filename`, `steering_angle`, `speed_kmh`, `brake`, `autonomous_mode`, `lane_side`, `line_detected`, `edge_detected`, `gps_x`, `gps_y`, `lidar_obstacle_detected`, `lidar_obstacle_distance`, `behavior_class`, `nav_command`, `obstacle_type`.

Set contributor ID via `$env:DATASET_CONTRIBUTOR_ID` before launching Webots.

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Webots Python API (controller, vehicle) is provided by the Webots installation.
#    Add the Webots lib/python directory to PYTHONPATH if needed.

# 3. Copy the controller to your Webots world's controller directory,
#    or point the robot node to the src/ path.
```

## Requirements

See `requirements.txt`. The Webots Python bindings (`controller`, `vehicle`) are **not** pip-installable — they ship with Webots R2023b or later.
