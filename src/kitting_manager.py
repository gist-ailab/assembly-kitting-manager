#!/usr/bin/env python

import json
import rospy
import cv2, cv_bridge
import numpy as np
import PIL
import message_filters

from std_msgs.msg import String, Header
from sensor_msgs.msg import RegionOfInterest, Image, CameraInfo, PointCloud2
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
import tf.transformations as tf_trans

import numpy as np
import copy
import time
import tf2_ros
import scipy as sp
import numpy.matlib as npm
import matplotlib.pyplot as plt

import open3d as o3d 
from open3d_ros_helper import open3d_ros_helper as orh
from utils import *

class KittingManager:

    def __init__(self):

        # initalize node
        rospy.init_node('kitting_manager')
        rospy.loginfo("Starting kitting_manager.py")

        self.params = rospy.get_param("kitting_manager")
        self.pad_factors = [1.2, 1.2, 1.2, 1.2]
        self.class_names = self.params["class_names"]
        self.idx2color = [[128, 255, 128], [13, 128, 255], [217, 12, 232], [232, 12, 128]]
        self.roi = self.params["roi"]
        self.bridge = cv_bridge.CvBridge()
        # subscribers
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("Waiting for transform for map to {}".format(self.params["camera_frame"]))
        # while True:
        #     try:
        #         self.transform_map_to_cam = self.tf_buffer.lookup_transform("map", self.params["camera_frame"], rospy.Time(), rospy.Duration(1.0))
        #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        #         rospy.sleep(0.5)
        #     else:
        #         self.H_map2cam = orh.msg_to_se3(self.transform_map_to_cam)
        #         rospy.loginfo("Sucessfully got transform from map to {}".format(self.params["camera_frame"]))
        #         break

        rospy.loginfo("Waiting for CenterMask client {}:{}".format(self.params["tcp_ip"], self.params["centermask_tcp_port"]))
        self.cm_sock, _ = su.initialize_server(self.params["tcp_ip"], self.params["centermask_tcp_port"])
        rospy.loginfo("Waiting for bracket client {}:{}".format(self.params["tcp_ip"], self.params["bracket_tcp_port"]))
        self.bracket_sock, _ = su.initialize_server(self.params["tcp_ip"], self.params["bracket_tcp_port"])
        # rospy.loginfo("Waiting for side client {}:{}".format(self.params["tcp_ip"], self.params["side_tcp_port"]))
        # self.side_sock, _ = su.initialize_server(self.params["tcp_ip"], self.params["side_tcp_port"])

        rgb_sub = message_filters.Subscriber(self.params["rgb"], Image, buff_size=2160*3840*3)
        point_sub = message_filters.Subscriber(self.params["point"], PointCloud2, buff_size=2160*3840*3)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, point_sub], queue_size=1, slop=1)
        self.ts.registerCallback(self.inference)
        self.camera_info = rospy.wait_for_message(self.params["camera_info"], CameraInfo)
        self.K = np.array(self.camera_info.K).reshape(3, 3)
        self.K_cropped = self.K
        self.K_cropped[0, 2] = self.K[0, 2] / 2
        self.K_cropped[1, 2] = self.K[1, 2] / 2

        # publishers
        self.vis_is_pub = rospy.Publisher('/assembly/vis_kitting', Image, queue_size=1)
        self.pose_array_pub = rospy.Publisher('/assembly/connector_pose', PoseArray, queue_size=1)


    def inference(self, rgb, pc_msg):
        ## 1. Get rgb, depth, point cloud
        start_time = time.time()
        camera_header = rgb.header
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        (x1, x2, y1, y2) = self.params["roi"]
        rgb_img = cv2.resize(np.uint8(rgb)[y1:y2, x1:x2], (self.params["width"], self.params["height"]), interpolation=cv2.INTER_LINEAR)
        cloud_cam = orh.rospc_to_o3dpc(pc_msg, remove_nans=True) 
        ## 2. Run centermask with client    
        su.sendall_image(self.cm_sock, rgb_img)
        vis_img = su.recvall_image(self.cm_sock)
        pred_masks = su.recvall_pickle(self.cm_sock) # [n_detect, 720, 1280] binary mask
        pred_boxes = su.recvall_pickle(self.cm_sock) # (x1, y1, x2, y2)
        pred_classes = su.recvall_pickle(self.cm_sock) # a vector of N confidence scores.
        pred_scores = su.recvall_pickle(self.cm_sock) # [0, num_categories).

        ## 3. Detect 2d pose
        poses_2d, vis_img = self.detect_2d_pose(rgb_img, vis_img, pred_classes, pred_boxes, pred_scores, pred_masks)
        self.vis_is_pub.publish(self.bridge.cv2_to_imgmsg(vis_img))
        
        ## 4. Convert 2d pose to 3d pose

        cloud_cam.estimate_normals()
        pose_3d_array = self.convert_2d_to_3d_pose(camera_header, poses_2d, cloud_cam)
        self.pose_array_pub.publish(pose_3d_array)
        


    def detect_2d_pose(self, rgb_img, vis_img, pred_classes, pred_boxes, pred_scores, pred_masks):
        boxes, scores, is_obj_ids, rgb_crops, is_masks = [], [], [], [], []
        poses_2d = []
        for itr, (label, (x1, y1, x2, y2), score, mask) in enumerate(zip(pred_classes, pred_boxes, pred_scores, pred_masks)):
            if score < self.params["is_thresh"]:
                continue
            # get largest contour for each instance
            mask = np.uint8(mask)
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key = cv2.contourArea)
            area = cv2.contourArea(contour)
            # ignore too small or large objects
            if area < 1e2 or 1e5 < area:
                continue
            vis_img = cv2.drawContours(vis_img.copy(), contours, 0, self.idx2color[label], 1)
            vis_img, cntr, p1, p2, angle = pca_analysis(contour, vis_img.copy())

            if self.params["class_names"][label+1] == "ikea_stefan_bracket":
                x1, x2, y1, y2 = get_bbox_offset(x1, x2, y1, y2, self.params["crop_offset"], self.params["width"], self.params["height"])
                rgb_crop = np.uint8(rgb_img[y1:y2, x1:x2].copy())
                # cv2.imwrite("/home/demo/rgb_{}.png".format(itr), rgb_crop)
                # rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (4,4))
                # rgb_crop = clahe.apply(rgb_crop)
                # rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_GRAY2RGB)
                # cv2.imwrite("/home/demo/clahe_{}.png".format(itr), rgb_crop)
                angle, p1, vis_img, is_stable = identify_bracket(self.bracket_sock, rgb_crop, vis_img.copy(), mask.copy(), cntr, p1, p2)
                text = "stable" if is_stable else "unstable"
                vis_img = cv2.putText(vis_img, text, (int(x1)-5, int(y2)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.idx2color[label], 2, cv2.LINE_AA)
            
            elif self.params["class_names"][label+1] == "ikea_stefan_bolt_side":
                # angle, p1, vis_img = identify_side(self.side_sock, rgb_img.copy(), self.params["width"], self.params["height"], self.params["crop_offset"], vis_img.copy(), mask.copy(), cntr, p1, p2)
                angle, p1, vis_img = identify_side(vis_img.copy(), mask.copy(), cntr, p1, p2)
            
            elif self.params["class_names"][label+1] == "ikea_stefan_bolt_hip":
                angle, p1, vis_img = identify_hip(vis_img.copy(), mask.copy(), cntr, p1, p2)
            
            elif self.params["class_names"][label+1] == "ikea_stefan_pin":
                angle, p1, vis_img = identify_pin(vis_img, cntr, p1)
            
            # visualize results
            vis_img = cv2.putText(vis_img, str(np.rad2deg(angle))[:3], (int(x1)-5, int(y2)+30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.idx2color[label], 2, cv2.LINE_AA)
            vis_img = cv2.putText(vis_img, self.params["class_names"][label+1].split('_')[-1], (int(x1)-5, int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, self.idx2color[label], 2, cv2.FONT_HERSHEY_SIMPLEX)
            vis_img = draw_axis(vis_img, cntr, p1, (0, 0, 255), 5)

            pose_2d = {"px": cntr[0], "py": cntr[1], "angle": angle}
            poses_2d.append(pose_2d)
        return poses_2d, vis_img

    def convert_2d_to_3d_pose(self, camera_header, poses_2d, cloud_cam):
        
        pose_3d_array = PoseArray()
        pose_3d_array.header = camera_header
        pose_3d_array.poses = []
        for pose_2d in poses_2d:
            start_time = time.time()
            px = pose_2d["px"]
            py = pose_2d["py"]
            angle = pose_2d["angle"]
            # get points xyz around px, py
            is_mask = np.zeros([self.params["original_height"], self.params["original_width"]])
            is_mask[py-self.params["pix_near"]:py+self.params["pix_near"], px-self.params["pix_near"]:px+self.params["pix_near"]] = 1
            cloud_center = orh.crop_with_2dmask(copy.deepcopy(cloud_cam), is_mask, self.K)
            cloud_center_npy = np.asarray(cloud_center.points)
            mask = (np.nan_to_num(cloud_center_npy) != 0).any(axis=1)
            cloud_center_npy = cloud_center_npy[mask]
            pos = np.median(cloud_center_npy, axis=0)
            if (np.isnan(pos)).any():
                rospy.logwarn("NaN position. consider to increase the thresh")
                continue
            # get normal vector around px, py
            # normal_center_npy = np.asarray(cloud_center.normals)
            # normal = np.median(normal_center_npy, axis=0)
            # # angle w.r.t image frame to normal 
            # v1 = np.array([0, -30, 1])
            # v2 = np.array([-30*math.cos(angle), 30*math.sin(angle), 1])
            # o = np.array([0, 0, 1])
            # V1 = np.matmul(np.linalg.inv(self.K_cropped), v1) * pos[-1]
            # V2 = np.matmul(np.linalg.inv(self.K_cropped), v2) * pos[-1]
            # O = np.matmul(np.linalg.inv(self.K_cropped), o) * pos[-1]
            # V1 = V1 - O
            # V2 = V2 - O
            # print(normal)
            # print(np.rad2deg(angle), vg.angle(V1, V2))
            # ori_2d = np.array([-30*math.cos(angle), 30*math.sin(angle), 1])
            # ori_3d = np.matmul(np.linalg.inv(self.K_cropped), ori_2d) * pos[-1]
            # if angle > np.pi:
                # angle - np.pi/2
            normal_rot = tf_trans.rotation_matrix(angle + np.deg2rad(self.params["angle_offset"]), (0, 0, 1))
            quat = tf_trans.quaternion_from_matrix(normal_rot)
            pose_3d_array.poses.append(orh.pq_to_pose(pos, quat))
        print("===")
        return pose_3d_array




if __name__ == '__main__':

    kitting_manager = KittingManager()
    rospy.spin()



