import os
import cv2
import time
import imageio
import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as msg_Image
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo


class RosD435CameraWrapper():
    def __init__(self, camera_name="cam_1", mode='fixed', calibration_result="."):
        rospy.Subscriber("/{}/color/image_raw".format(camera_name), msg_Image, self.rgb_callback1)
        rospy.Subscriber("/{}/aligned_depth_to_color/image_raw".format(camera_name), msg_Image, self.depth_callback1)
        rospy.Subscriber("/{}/aligned_depth_to_color/camera_info".format(camera_name), CameraInfo, self.cameraInfo_callback) 

        self.rgb_image1 = None
        self.depth_image1 = None
        self.K = None

        self.record_flag = False
        self.rgb_list = []
        self.depth_list = []
        self.mode = mode

        if mode == 'fixed':
            self.cam2base = np.load(os.path.join(calibration_result, 'cam2base.npy'))

        if mode == 'inhand':
            self.cam2gripper = np.load(os.path.join(calibration_result, 'cam2gripper.npy'))

    def rgb_callback1(self, data):
        bridge = CvBridge()
        # self.rgb_image1 = cv2.cvtColor(cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480)), cv2.COLOR_RGB2BGR)
        self.rgb_image1 = cv2.cvtColor(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), cv2.COLOR_RGB2BGR)

        if self.record_flag:
            self.rgb_list.append(self.rgb_image1)

            # cv2.namedWindow("record_window")
            # cv2.imshow("record_window", self.rgb_image1)
            # key = cv2.waitKey(1)

            # if key & 0xFF == ord('b'):
            #     cv2.destroyWindow('record_window')
            #     self.record_stop()

    def depth_callback1(self, data):
        bridge = CvBridge()
        # self.depth_image1 = cv2.resize(np.array(bridge.imgmsg_to_cv2(data, data.encoding)), (640, 480))
        self.depth_image1 = np.array(bridge.imgmsg_to_cv2(data, data.encoding))
        
        if self.record_flag:
            self.depth_list.append(self.depth_image1)

    def cameraInfo_callback(self, data):
        self.K = np.array(data.K).reshape((3, 3)) 

    def get_images(self, rotate_180=False):
        if rotate_180:
            rgb_image1 = cv2.rotate(self.rgb_image1, cv2.ROTATE_180)
            depth_image1 = cv2.rotate(self.depth_image1, cv2.ROTATE_180)

            return [rgb_image1, depth_image1]
        else:
            return [self.rgb_image1, self.depth_image1]

    def record_start(self):
        print("record start....")
        self.record_flag = True

        return self.record_flag
    
    def record_pause(self):
        print("record pause....")
        self.record_flag = False
        # cv2.destroyWindow('record_window')
    
    def record_stop(self, video_save_path):
        print("record stop....")
        self.record_flag = False
        out_list = []

        if len(self.rgb_list) != 0:
            for i, img in enumerate(self.rgb_list):
                if i % 3 == 0:
                    out_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            imageio.mimwrite(video_save_path, out_list, quality=8, fps=20)
            print("Success save video to {}".format(video_save_path))

    def get_intrinsic(self):
        return self.K
        
    def get_cam2base(self, curr_gripper_pos=None, curr_gripper_ori=None):

        if self.mode == 'fixed':
            return self.cam2base
        
        if self.mode == 'inhand':
            if curr_gripper_pos is None:
                print("Camera inhand, please provide current gripper pose to get cam2base !!!")
                exit()

            
            # curr_gripper_ori is expected in [x, y, z, w] (xyzw) order
            q = np.array(curr_gripper_ori, dtype=float)
            if q.shape[0] != 4:
                raise ValueError("Quaternion must be length 4")
            gripper2base = np.eye(4)
            #gripper2base[:3, :3] = Rot.from_quat(q[[1,2,3,0]].copy()).as_matrix()
            gripper2base[:3, :3] = Rot.from_quat(q).as_matrix()
            gripper2base[:3, 3] = curr_gripper_pos.copy()
            cam2base = gripper2base @ self.cam2gripper

            return gripper2base, cam2base
    
    def get_object2base(self, camera2base, object2camera_pos, object2camera_ori):
        object2camera = np.eye(4)
        # object2camera_ori is expected in [x, y, z, w] (xyzw) order
        q = np.array(object2camera_ori, dtype=float)
        if q.shape[0] != 4:
            raise ValueError("Quaternion must be length 4")
            
        #object2camera[:3, :3] = Rot.from_quat(q).as_matrix()
        object2camera[:3, :3] = np.eye(3) # DOPE no rotation
        object2camera[:3, 3] = object2camera_pos.copy()

        # Popo's math correction
        correction_popo1 = np.array([1, 0, 0, 0], dtype=float)
        correction_popo2 = np.array([0, np.sqrt(2)/2, 0, np.sqrt(2)/2], dtype=float)

        camera2base[:3, :3] = camera2base[:3, :3] @ Rot.from_quat(correction_popo1).as_matrix() @ Rot.from_quat(correction_popo2).as_matrix()

        object2base = camera2base @ object2camera

        return object2base

# import message_filters


# class RosCameraWrapper():
#     def __init__(self):
#         imageL = message_filters.Subscriber("/cam_1/color/image_raw", msg_Image)
#         imageR = message_filters.Subscriber("/cam_2/color/image_raw", msg_Image)

#         # Synchronize images
#         ts = message_filters.ApproximateTimeSynchronizer([imageL, imageR], queue_size=10, slop=0.5)
#         ts.registerCallback(self.image_callback)
        
#     def image_callback(self, imageL, imageR):
#         brige = CvBridge()
#         rospy.loginfo("receiving frame")
#         imageLeft = brige.imgmsg_to_cv2(imageL)
#         imageRight = brige.imgmsg_to_cv2(imageR)
        
#         self.rgb_image = [imageLeft, imageRight]

#     def get_images(self):
#         return self.rgb_image