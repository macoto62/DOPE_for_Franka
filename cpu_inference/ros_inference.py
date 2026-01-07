import argparse
import time
import cv2
import numpy as np
import os
import json
import yaml
import onnxruntime as ort
import subprocess
import sys

# from catkin_ws.src.franka_human_friendly_controllers.python.LfD import camera
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_matrix, quaternion_from_euler
import rospy
from Learning_from_demonstration import LfD
from pose_transform_functions import  array_quat_2_pose


sys.path.append("../common/")
from PIL import Image as PILImage
from cuboid import Cuboid3d
from detector import ObjectDetector
from cuboid_pnp_solver import CuboidPNPSolver
from utils import loadimages_inference, Draw

from camera import RosFemtoCameraWrapper
from camera_d435 import RosD435CameraWrapper

curr_pos = None
curr_ori = None

class ONNXDopeNode:
    def __init__(self, config, model_path, class_name):
        self.input_is_rectified = config["input_is_rectified"]
        self.downscale_height = config["downscale_height"]
        
        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config["thresh_angle"]
        self.config_detect.thresh_map = config["thresh_map"]
        self.config_detect.sigma = config["sigma"]
        self.config_detect.thresh_points = config["thresh_points"]

        self.loc = None
        self.ori = None
        self.score = None
        # Create the ONNX Runtime session with the desired execution provider
        # (do not pass providers to run(); set them at session creation)
        try:
            self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except TypeError:
            # Older versions of onnxruntime may not accept the providers kwarg here.
            # Fallback to creating a session without explicit providers.
            self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        
        try:
            self.draw_color = tuple(config["draw_colors"][class_name])
        except:
            self.draw_color = (0, 255, 0)

        self.dimension = tuple(config["dimensions"][class_name])
        self.class_id = config["class_ids"][class_name]
        self.pnp_solver = CuboidPNPSolver(class_name, cuboid3d=Cuboid3d(config["dimensions"][class_name]))
        self.class_name = class_name

    def preprocess_image(self, img):
        #img = cv2.resize(img, (640,480 ))  # Adjust to match ONNX input
        img = img.astype(np.float32) / 255.0  # Normalize 
        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        #img = (float)(img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W)
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, C, H, W)
        return img

    def infer(self, img):
        input_tensor = self.preprocess_image(img)
        # Call run without the 'providers' kwarg; providers must be set on the session
        outputs = self.ort_session.run(["351", "364"], {self.input_name: input_tensor})
        return outputs

    def image_callback(self, img, camera_info, img_name, output_folder, debug=False):
        P = np.matrix(camera_info["projection_matrix"]["data"], dtype="float32").copy()
        P.resize((3,4))
        camera_matrix = P [:, :3]
        dist_coeffs = np.zeros((4, 1))
        
        # trim() function, I want to original image (480, 560) trim right and left to 480
        img = img[:, 40:520, :] 
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img = img[:, 80:560, :]

        self.score = None

        print(img.shape)
        height, width, _ = img.shape # [512, 512, 3] 
        height_scaling_factor = float(self.downscale_height) / height
        width_scaling_factor = float(self.downscale_height) / width
        if height_scaling_factor < 1.0:
            camera_matrix[0,0] *= width_scaling_factor
            camera_matrix[0,2] *= width_scaling_factor
            camera_matrix[1,1] *= height_scaling_factor
            camera_matrix[1,2] *= height_scaling_factor
            img = cv2.resize(img, (int(width_scaling_factor * width), int(height_scaling_factor * height)))
            print(img.shape)
        
        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs)

        img_copy = img.copy()
        #img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        im = PILImage.fromarray(img_copy)
        draw = Draw(im)

        dict_out = {"camera_data": {}, "objects": []}
        outputs = self.infer(img)
        #print(len(outputs))
        vertex2 = outputs[0][0]
        aff = outputs[1][0]

        
        results, belief_imgs = ObjectDetector.detect_object_in_image(
            vertex2,aff, self.pnp_solver, img, self.config_detect,
            grid_belief_debug=debug
        )
        
        for _, result in enumerate(results):
            if result["location"] is None:
                continue

            loc = result["location"]
            ori = result["quaternion"]
            # score = result["score"]
            self.loc = loc
            self.ori = ori
            # self.score = score

            dict_out["objects"].append(
                {
                    "class": self.class_name,
                    "location": np.array(loc).tolist(),
                    "quaternion_xyzw": np.array(ori).tolist(),
                    "projected_cuboid": np.array(result["projected_points"]).tolist(),
                }
            )

            # Draw the cube
            if None not in result["projected_points"]:
                points2d = []
                for pair in result["projected_points"]:
                    points2d.append(tuple(pair))
                draw.draw_cube(points2d, self.draw_color)

        # create directory to save image if it does not exist
        img_name_base = img_name.split("/")[-1]
        output_path = os.path.join(
            output_folder,
            *img_name.split("/")[:-1],
        )
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        im.save(os.path.join(output_path, img_name_base))
        if belief_imgs is not None:
            belief_imgs.save(os.path.join(output_path, "belief_maps.png"))

        json_path = os.path.join(
            output_path, ".".join(img_name_base.split(".")[:-1]) + ".json"
        )
        # save the json files
        with open(json_path, "w") as fp:
            json.dump(dict_out, fp, indent=2)

# add robot move from popo1021.py
def demo_workflow_initial_move_by_popo():

    # Initialize
    lfd = LfD()
    lfd.home()
    rospy.sleep(5.0)
    # lfd.traj_rec()
    
    #lfd.set_stiffness(1000, 1000, 1000, 30, 30, 30, 50)
    rospy.loginfo("Popoing...")
    print("open gripper first")
    rospy.loginfo("Opening gripper to release object...")
    lfd.grasp_gripper(width=0.08, force=5.0)
    rospy.sleep(3.0)

    goal = PoseStamped()
    goal.header.frame_id = "panda_link0"

    pos_array = np.array([0.39845, 0.16455, 0.51202])  # x,y,z
    quat = np.quaternion(0,0.99258,0.02255,0.11937)  # w,x,y,z
    goal = array_quat_2_pose(pos_array, quat)
    goal.header.seq = 1
    goal.header.stamp = rospy.Time.now()
    print("Moving to target pose using Cartesian control...")
    lfd.go_to_pose(goal)
    print("Reached target pose.")

def grab_object_by_popo(popo_x, popo_y):

    # Initialize
    lfd = LfD()
    lfd.home()
    rospy.sleep(5.0)
    # lfd.traj_rec()
    
    #lfd.set_stiffness(1000, 1000, 1000, 30, 30, 30, 50)
    rospy.loginfo("Popoing...")
    print("open gripper first")
    rospy.loginfo("Opening gripper to release object...")
    lfd.grasp_gripper(width=0.08, force=5.0)
    rospy.sleep(3.0)

    goal = PoseStamped()
    goal.header.frame_id = "panda_link0"
    popo_x = popo_x - 0.01
    popo_y = popo_y + 0.026

    pos_array = np.array([popo_x, popo_y, 0.25])  # x,y,z
    quat = np.quaternion(0,1,0,0)  # w,x,y,z
    goal = array_quat_2_pose(pos_array, quat)
    goal.header.seq = 1
    goal.header.stamp = rospy.Time.now()
    print("Moving to target pose using Cartesian control...")
    lfd.go_to_pose(goal)
    print("Reached target pose.")

# SCP to laptop B after inference successfulla
local_file = "soup_location.npy"
remote_user = "caslab"
remote_ip = "192.168.0.130"
remote_path_dir = "/home/caslab/Desktop/nody_robot/tmp" 

def scp_to_laptopB(target_x, target_y, target_z):
    try:
        # 1. 將 x, y, z 寫入本地的 npy 檔案
        # 將數據打包成 numpy array
        data = np.array([target_x, target_y, target_z])
        np.save(local_file, data)
        print(f"[Local] {local_file} saved successfully (Data: {data}).")

        # 2. 組合 SCP 指令
        # 格式: scp local_file user@ip:remote_directory
        destination = f"{remote_user}@{remote_ip}:{remote_path_dir}"
        
        cmd = [
            "scp",
            local_file,
            destination
        ]

        # 3. 執行 SCP 指令
        print(f"[SCP] Sending to {destination}...")
        subprocess.run(cmd, check=True)
        print("[SCP] Transfer completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"[Error] SCP transfer failed: {e}")
    except Exception as e:
        print(f"[Error] An error occurred: {e}")
    
def rotation_matrix_to_quaternion(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S = 4 * qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S = 4 * qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S = 4 * qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

def pose_to_list(pose):
    # pose: geometry_msgs/Point or similar with .x/.y/.z
    if pose is None:
        return None
    return [float(pose.x), float(pose.y), float(pose.z)]

def quat_to_list(quat):
    # quat: geometry_msgs/Quaternion with .x/.y/.z/.w
    if quat is None:
        return None
    return [float(quat.x), float(quat.y), float(quat.z), float(quat.w)]

# numpy versions
def pose_to_numpy(pose):
    lst = pose_to_list(pose)
    return None if lst is None else np.array(lst, dtype=float)

def quat_to_numpy(quat):
    lst = quat_to_list(quat)
    return None if lst is None else np.array(lst, dtype=float)

def ee_pos_callback(msg):
    global curr_pose, curr_ori
    curr_pose = msg.pose.position
    curr_ori = msg.pose.orientation

def image_callback(msg):
    global curr_pose, curr_ori, image_count
    # Stop saving if we already captured enough images this cycle
    if image_count >= 1:
        return

    if curr_pose is None:
        return  # skip until we have a pose

    # Robustly handle different camera encodings (rgb8, bgr8, etc.).
    # Keep an in-memory RGB image (for model inference) and save a BGR image
    # on disk (cv2.imwrite expects BGR ordering).
    bgr_to_save = cv2.cvtColor(np.array(bridge.imgmsg_to_cv2(msg, msg.encoding)), cv2.COLOR_RGB2BGR)

    # rgb_img = None
    # try:
    #     enc = getattr(msg, 'encoding', '').lower()
    #     if 'rgb' in enc:
    #         # message already RGB; request rgb8 to get an RGB numpy array
    #         rgb_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    #     elif 'bgr' in enc:
    #         # message already BGR; request bgr8 then convert to RGB in memory
    #         bgr_tmp = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         rgb_img = bgr_tmp[..., ::-1].copy()
    #     else:
    #         # Unknown encoding: try rgb8 first, then bgr8 fallback
    #         try:
    #             rgb_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    #         except Exception:
    #             bgr_tmp = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #             rgb_img = bgr_tmp[..., ::-1].copy()
    # except Exception as e:
    #     rospy.logwarn(f"Failed to convert ROS Image to CV image: {e}")
    #     return

    # Prepare BGR image for saving with OpenCV (cv2.imwrite)
    # try:
    #     bgr_to_save = rgb_img[..., ::-1].copy()
    # except Exception as e:
    #     rospy.logwarn(f"Failed to prepare BGR image for saving: {e}")
    #     return

    # Save image (on-disk images will be BGR so cv2.imread later returns BGR)
    filename = f"../femto_data/frame_{image_count:d}.jpg"
    try:
        cv2.imwrite(filename, bgr_to_save)
    except Exception as e:
        rospy.logwarn(f"Failed to write image {filename}: {e}")
        return

    # Save corresponding pose (unchanged format)
    try:
        with open("../femto_data/pose_log.txt", "a") as f:
            f.write(f"{filename}, "
                    f"{curr_pose.x:.8f}, {curr_pose.y:.8f}, {curr_pose.z:.8f}, "
                    f"{curr_ori.x:.8f}, {curr_ori.y:.8f}, {curr_ori.z:.8f}, {curr_ori.w:.8f}\n")
    except Exception as e:
        rospy.logwarn(f"Failed to append pose to log: {e}")

    rospy.loginfo(f"Saved {filename}")
    image_count += 1

def initialize_subscribers():
    """Initialize ROS subscribers once at the start."""
    global sub_pose, sub_image, curr_pose, curr_ori
    
    print("Initializing subscribers...")
    sub_pose = rospy.Subscriber("/cartesian_pose", PoseStamped, ee_pos_callback)
    sub_image = rospy.Subscriber("/cam_1/color/image_raw", ROSImage, image_callback)
    
    rospy.sleep(0.1)  # Allow time to establish connections
    
    # Ensure we have at least one pose message
    try:
        if curr_pose is None:
            rospy.loginfo("Waiting for first /cartesian_pose message...")
            msg = rospy.wait_for_message("/cartesian_pose", PoseStamped, timeout=5.0)
            curr_pose = msg.pose.position
            curr_ori = msg.pose.orientation
            print("Initial pose received.")
    except Exception as e:
        rospy.logwarn(f"Timeout or error waiting for /cartesian_pose: {e}")
    
    print("Subscribers initialized and ready.")

def cleanup_subscribers():
    """Unregister all ROS subscribers."""
    global sub_pose, sub_image
    
    print("Cleaning up subscribers...")
    if sub_image is not None:
        try:
            sub_image.unregister()
            print("Unregistered image subscriber")
        except Exception as e:
            print(f"Error unregistering image subscriber: {e}")
    
    if sub_pose is not None:
        try:
            sub_pose.unregister()
            print("Unregistered pose subscriber")
        except Exception as e:
            print(f"Error unregistering pose subscriber: {e}")

def take_photos():
    """Capture images for a fixed duration using active subscribers.
    
    This function assumes subscribers are already active and just waits
    for the capture duration to allow image_callback to save images.
    """
    global image_count
    image_count = 0
    
    # Capture images for a fixed duration
    duration_s = 0.005
    rospy.loginfo(f"Taking photos for {duration_s} seconds...")
    start_t = time.time()
    try:
        while time.time() - start_t < duration_s and not rospy.is_shutdown():
            rospy.sleep(0.01)
    except Exception:
        pass

    rospy.loginfo(f"Photo capture window complete. Captured {image_count} images.")

    # cmd = [
    #     "rosrun",
    #     "image_view",
    #     "image_saver",
    #     "image:=/femto_1/color/image_raw",
    #     "_filename_format:=../femto_data/frame_%04d.jpg",
    # ]

    # # Start the saver process
    # try:
    #     proc = subprocess.Popen(cmd)
    # except Exception as e:
    #     print(f"Failed to start image_saver: {e}")
    #     return

    # # Wait for up to 3 seconds for captures
    # try:
    #     proc.wait(timeout=1.0)
    #     # If the process exited early that's fine
    # except subprocess.TimeoutExpired:
    #     # Try graceful shutdown first
    #     proc.terminate()
    #     try:
    #         proc.wait(timeout=1.0)

    #     except subprocess.TimeoutExpired:
    #         proc.kill()

    # # Ensure no leftover image_saver processes remain
    # try:
    #     subprocess.call(["pkill", "-9", "-f", "image_saver"])
    # except Exception:
    #     pass

    print("Photos taken and image_saver stopped.")

def remove_pictures_in_folder(folder="../femto_data"):
    global image_count
    image_count = 0
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    print("deleted all pictures in folder ", folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outf", default="output")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="../config/config_pose.yaml")
    parser.add_argument("--camera", default="../config/camera_info.yaml")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--object", required=True)
    opt = parser.parse_args()
    
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)

    rospy.init_node('onnx_dope_inference_node')
    rate = rospy.Rate(30)  # 30 Hz  
    bridge = CvBridge()
    image_count = 0
    imgs, imgsname = loadimages_inference(opt.data, extensions=["png", "jpg"])

    frame_number = 0
    remove_pictures_in_folder("./output")
    remove_pictures_in_folder("../femto_data")

    # demo_workflow_initial_move_by_popo()

    try:
        # Initialize subscribers once at the beginning
        initialize_subscribers()
        
        while not rospy.is_shutdown():
            is_detected = False
            # no picture or object not detected yet
            while not is_detected: 
                print("Capturing photos...")
                # Need to like photo_taken.py to write the end effector pose along with images
                take_photos()
                imgs, imgsname = loadimages_inference(opt.data, extensions=["png", "jpg"])
                dope_node = ONNXDopeNode(config, opt.model, opt.object)
                # Inference from the images
                for i, img_path in enumerate(imgs):
                    cur_time=time.time()
                    print(img_path)
                    frame = cv2.imread(img_path)
                    #frame = frame[..., ::-1].copy()
                    dope_node.image_callback(frame, camera_info, imgsname[i], opt.outf)
                    print(f"CPU_runtime: {time.time()-cur_time}") 
                    # if dope_node.score is not None:
                    #     print("Confidence score: ", dope_node.score)
                    # else:
                    #     print("Confidence score: None (no object detected)")
                    # dope_node.loc is cam2object position
                    # dope_node.ori is cam2object orientation
                    if dope_node.loc is not None:
                        is_detected = True
                        print(f"Detected object at location: {dope_node.loc}, orientation: {dope_node.ori}")
                        frame_number = i
                        break  # Exit inner loop if detection is successful
                if not is_detected:
                    print("Object not detected, taking more photos...")
                    remove_pictures_in_folder("./output")
                    remove_pictures_in_folder("../femto_data")
            if is_detected:
                # Clean up subscribers since object is detected
                cleanup_subscribers()
                print("Object detected successfully!")
                break
        print("Finished inference loop")
        print(f"Using frame number: {frame_number}")

        # derive the image filename and try to read the corresponding gripper pose
        image_fname = f"../femto_data/frame_{frame_number:d}.jpg"
        pose_log = "../femto_data/pose_log.txt"

        gripper_pos = None
        gripper_ori = None
        try:
            with open(pose_log, "r") as pf:
                for line in pf:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 8:
                        continue
                    fname = parts[0]
                    # match by basename or full path
                    if os.path.basename(fname) == os.path.basename(image_fname) or fname == image_fname:
                        vals = [float(x) for x in parts[1:8]]
                        gripper_pos = np.array(vals[0:3], dtype=float)
                        gripper_ori = np.array(vals[3:7], dtype=float)
                        print(f"Found gripper pose for image {image_fname}: pos {gripper_pos}, ori {gripper_ori}")
                        break
        except FileNotFoundError:
            print(f"Pose log not found at {pose_log}")

        if gripper_pos is None:
            print(f"Could not find pose for image {image_fname} in {pose_log}")
        else:
            # instantiate camera wrapper; prefer 'inhand' if calibration available
            camera_wrapper = RosD435CameraWrapper(camera_name="cam_1", mode="inhand")

            # get camera-to-base transform
            try:
                # get_cam2base accepts curr_gripper_pos and curr_gripper_ori when inhand
                gripper2base, cam2base = camera_wrapper.get_cam2base(curr_gripper_pos=gripper_pos, curr_gripper_ori=gripper_ori)
            except Exception:
                # fallback to fixed cam2base
                gripper2base, cam2base = camera_wrapper.get_cam2base()
            
            cam2base[:3, :3] = gripper2base[:3,:3]

            print("Gripper to base pose:\n", gripper2base)
            print("Camera to base pose:\n", cam2base)
        
            # load the inference JSON produced earlier to get camera->object pose
            json_candidates = [
                os.path.join(opt.outf, f"frame_{frame_number:d}.json"),
                os.path.join(opt.outf, os.path.basename(image_fname).replace('.jpg', '.json')),
            ]
            json_path = None
            for p in json_candidates:
                if os.path.exists(p):
                    json_path = p
                    break

            if json_path is None:
                # try matching any json in output that contains the image base name
                for root, _, files in os.walk(opt.outf):
                    for f in files:
                        if f.endswith('.json') and f.startswith(f"frame_{frame_number:d}"):
                            json_path = os.path.join(root, f)
                            break
                    if json_path:
                        break

            if json_path is None:
                print(f"Could not find output json for frame {frame_number} in {opt.outf}")
            else:
                with open(json_path, 'r') as jf:
                    data = json.load(jf)

                # select first object (or the one matching class)
                obj_entry = None
                for obj in data.get('objects', []):
                    # if object class matches requested, pick it
                    if obj.get('class') == opt.object:
                        obj_entry = obj
                        break
                if obj_entry is None and len(data.get('objects', [])) > 0:
                    obj_entry = data['objects'][0]

                if obj_entry is None:
                    print(f"No detected objects in {json_path}")
                else:
                    object2camera_pos = np.array(obj_entry['location'], dtype=float)
                    object2camera_ori = np.array([1,0,0,0], dtype=float)  # default orientation
                    #object2camera_ori = np.array(obj_entry.get('quaternion_xyzw', [0, 0, 0, 1]), dtype=float)
                    
                    #print("Object orientation (quat xyzw): ", object2camera_ori)
                    # change coordinate system by Popo
                    object2camera_pos /= 100.0 # from cm to m
                    object2camera_pos = np.array([object2camera_pos[2], -object2camera_pos[0], -object2camera_pos[1]])
                    print("Object to camera position (m): ", object2camera_pos)
                    # compute object->base
                    object2base_pos_from_gripper = camera_wrapper.get_object2base(gripper2base, object2camera_pos, object2camera_ori)
                    object2base_pos = camera_wrapper.get_object2base(cam2base, object2camera_pos, object2camera_ori)
                    print("Object->base from gripper (4x4):")
                    print(object2base_pos_from_gripper)
                    print("Object->base transform from camera (4x4):")
                    print(object2base_pos)
                    target_pos = object2base_pos[0:3,3]
                    target_pos_from_gripper = object2base_pos_from_gripper[0:3,3]
                    print("Target position in base frame: ", target_pos_from_gripper)
                    target_quat = rotation_matrix_to_quaternion(object2base_pos_from_gripper[0:3,0:3])
                    print("Target orientation in base frame: ", target_quat)
                    print("Target position from camera, target position: ", target_pos)

                    # print("\n\n\n Are you sure to grab the object at position above? \n\n\n")
                    # wait_for_input = input("Press Enter to continue...")
                    # if(wait_for_input.lower() in ['']):
                    #     grab_object_by_popo(target_pos[0], target_pos[1])
                    # else:
                    #     print("Exiting without grabbing.")

                    # Finally, SCP a log file to laptop B to signal success
                    target_pos[1] += 0.025
                    scp_to_laptopB(target_pos[0], target_pos[1], 0.5)
                    

        # fetch "../femto_data/frame_{frame_number:d}.jpg" and its corresponding pose from pose_log.txt
        # end_popo = ../femto_data/frame_{frame_number:d}.jpg" and its corresponding pose
        # # this is the cartesian pose of the end effector when the picture was taken
        # camera_wrapper = RosFemtoCameraWrapper()
        # camera_popo = camera_wrapper.get_cam2base(self, curr_gripper_pos=end_popo.pos, curr_gripper_ori=end_popo.ori)

        # # then fetch "output/frame_{frame_number:d}.json" to get cam2object pose
        # cam2object_popo = camera_wrapper.get_object2base(camera_popo, dope_node.loc, dope_node.ori)
        # final_pos = get_object2base(camera2base_pos, object2camera_pos, object2camera_ori)
        
        #print("Object to base position:\n", object2base_pos)
        #panda: goto_position(object2base_pos) height + 30 cm
        #gripper movement to grasp
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by ROS")
    except Exception as e:
        print(f"An error occurred during inference: {e}")

