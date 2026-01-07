Prompt
```
I use Franka panda emika robotics right now, equipped with the camera on the gripper (in-hand),
what I try to do is to use the DOPE(6-dof model) to capture the object.
And right now, I can get the pos+quarternion of the gripper, and use hand-eye calibration matrix to get the camera pos and quarternion
the camera pos seems correct, but I don't know whether the camera orientation is correct.
And I get the 6-dof model from femto_1 camera, then try to get the object relative to the base, but it seems that I always get the larger front number of the global
what would be the problem and how to fix it? 
```

This is Readme file
#ONNX MODEL
dope_network_400.onnx

# Test the onnx model
cpu_run_onnx.py: Success
gpu_run_onnx.py: Failed

# RUN DOPE model
inference.py: provided by the NVidia
dope_inference.py: RUN SUCCESSFULLY
python3 dope_inference.py --data ../sample_data --model dope_network_400.onnx --object cracker

# Project
cd ~/catkin_ws/src/franka_human_friendly_controllers/python/LfD/onnxruntime-raspberrypi/cpu_inference
source ../cpu_env/bin/activate
python3 ros_inference.py --data ../femto_data --model soup_dope_network.onnx --object soup
