import argparse
import time
import cv2
import numpy as np
import os
import json
import yaml
import onnxruntime as ort
import sys
sys.path.append("../common/")
from PIL import Image
from cuboid import Cuboid3d
from detector import ObjectDetector
from cuboid_pnp_solver import CuboidPNPSolver
from utils import loadimages_inference, Draw

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
        im = Image.fromarray(img_copy)
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
            
def take_photos():
    now = time.time()
    while time.time() - now < 2.0:
        os.system("rosrun image_view image_saver image:=/femto_1/color/image_raw _filename_format:='../femto_data/frame_%04d.jpg'")
    time.sleep(0.1)
    print("Photos taken.")

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
    
    imgs, imgsname = loadimages_inference(opt.data, extensions=["png", "jpg"])

    dope_node = ONNXDopeNode(config, opt.model, opt.object)
    for i, img_path in enumerate(imgs):
        cur_time=time.time()
        print(img_path)
        frame = cv2.imread(img_path)
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = frame[..., ::-1].copy()
        dope_node.image_callback(frame, camera_info, imgsname[i], opt.outf)
        print(f"CPU_runtime: {time.time()-cur_time}") 

    # # take some photos
    # take_photos()
    # imgs, imgsname = loadimages_inference(opt.data, extensions=["png", "jpg"])
