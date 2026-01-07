# DOPE for Franka: Soup Can Pose Estimation

This repository integrates NVIDIA’s Deep Object Pose (DOPE) with the Franka Emika Panda robotic arm using the "Franka Human Friendly Controllers" framework (https://github.com/franzesegiovanni/franka_human_friendly_controllers). It provides CPU ONNX inference pipelines to detect and estimate the 6D pose of a target object (a soup can) and includes ROS utilities and Learning-from-Demonstration (LfD) scripts to enable manipulation tasks.

## Introduction
- Based on: https://github.com/NVlabs/Deep_Object_Pose (DOPE)
- Robotics integration: [Franka Emika Panda via franka_human_friendly_controllers](https://github.com/franzesegiovanni/franka_human_friendly_controllers.git)
- Objective: Detect the soup can and estimate its 6D pose to enable grasping and manipulation.

## Features
- Real-time 6D pose estimation of a soup can using DOPE ONNX models.
- CPU inference pipelines with ONNX Runtime.
- ROS integration for camera streaming and publishing detections.
- Camera calibration and hand–eye (camera-to-gripper) transform support.
- LfD scripts to execute simple task sequences on the Franka arm.

## Requirements
- OS: Ubuntu 20.04 with RTOS kernel
- ROS1
- Hardware:
  - Franka Emika Panda 
  - RGB camera (e.g., Intel RealSense D435 or Femto Bolt)

## Models
This repo references ONNX models for DOPE (e.g., `cpu_inference/soup_dope_network.onnx`). You can download from this [link](https://drive.google.com/drive/u/0/folders/1BK8Txil5Hfk3p2Ambrw_W44orfs9NeDF)

## Quick Start

### Install ROS
https://wiki.ros.org/Installation/Ubuntu
### CPU inference (offline/demo)
Run DOPE inference on a connected camera or sample frames.
```
pip install -r requirements.txt 
python3 ros_inference.py --data ../femto_data --model soup_dope_network.onnx --object soup
```
Results are saved to `cpu_inference/output/` (e.g., JSON keypoints/poses).


## Franka Integration
- Franka helpers in `cpu_inference/panda.py` bridge detections to robot actions.
- The LfD script `cpu_inference/Learning_from_demonstration.py` demonstrates how to chain detection + motion to perform a simple pick/place with the soup can.

## Configuration
- Edit `config/camera_info.yaml` for intrinsics.
- Tune thresholds in `common/detector.py` and related files (e.g., confidence thresholds).
- For custom objects: Train/convert your DOPE model to ONNX, then place it under `cpu_inference/` and update scripts accordingly.

## Outputs
- JSON pose/keypoint files are written to `cpu_inference/output/`.
- You can visualize or consume these in downstream planning or control modules.

## Future Work
1. Add a robust GPU inference version (CUDA/TensorRT-backed ONNX Runtime).
2. Replace ad-hoc image capture with ROS-native pipelines for performance and sustainability.
3. Evaluate stronger models (e.g., FoundationPose) or simpler 3-DoF/2D detection depending on task needs.
4. Improve developer experience: containerize with Docker and set up reliable CI/CD (linting, tests, build).

## Attribution
- DOPE: NVLabs — Deep Object Pose (https://github.com/NVlabs/Deep_Object_Pose)
- Franka Human Friendly Controllers: integration layer used to control the Panda arm.

## License
This project builds on DOPE and related works. Refer to the upstream licenses and ensure compliance when distributing trained weights and models.
