#!/usr/bin/env python

import json
import rospy
import cv2, cv_bridge
import numpy as np
import PIL
import message_filters

from std_msgs.msg import String, Header
from sensor_msgs.msg import RegionOfInterest, Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from assembly_msgs.srv import GetObjectPoseArray
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
        self.class_names = self.params["class_names"]
        self.idx2color = [[42, 42, 165], [13, 128, 255], [217, 12, 232], [232, 12, 128]]
        self.roi = self.params["roi"]
        self.bridge = cv_bridge.CvBridge()
        # subscribers
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("Waiting for transform for map to {}".format(self.params["camera_frame"]))
        while True:
            try:
                self.transform_map_to_cam = self.tf_buffer.lookup_transform("map", self.params["camera_frame"], rospy.Time(), rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.sleep(0.5)
            else:
                self.H_map2cam = orh.msg_to_se3(self.transform_map_to_cam)
                rospy.loginfo("Sucessfully got transform from map to {}".format(self.params["camera_frame"]))
                break

        rospy.loginfo("Waiting for CenterMask client {}:{}".format(self.params["tcp_ip"], self.params["centermask_tcp_port"]))
        self.cm_sock, _ = su.initialize_server(self.params["tcp_ip"], self.params["centermask_tcp_port"])
        rospy.loginfo("Waiting for bracket client {}:{}".format(self.params["tcp_ip"], self.params["bracket_tcp_port"]))
        self.bracket_sock, _ = su.initialize_server(self.params["tcp_ip"], self.params["bracket_tcp_port"])
        rospy.loginfo("Waiting for side client {}:{}".format(self.params["tcp_ip"], self.params["side_tcp_port"]))
        self.side_sock, _ = su.initialize_server(self.params["tcp_ip"], self.params["side_tcp_port"])

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
        self.objectposes_srv = rospy.Service('/get_object_pose_array', GetObjectPoseArray, self.get_object_pose_array)

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
        # sort pose_2d with score
        if len(poses_2d) > 0:
            poses_2d = sorted(poses_2d, key=lambda k: k['score'], reverse=True)
            x1, x2, y1, y2 = poses_2d[0]["bbox"]
            vis_img = cv2.putText(vis_img, "Best Grasp", (int(x1)-15, int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 13, 255), 2, cv2.LINE_AA)
            vis_img = cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (13, 13, 255), 3)
        self.vis_is_pub.publish(self.bridge.cv2_to_imgmsg(vis_img))
        
        ## 4. Convert 2d pose to 3d pose
        cloud_cam.estimate_normals()
        header = camera_header
        header.frame_id = "map"
        pose_3d_array = self.convert_2d_to_3d_pose(header, poses_2d, cloud_cam)
        self.pose_array_pub.publish(pose_3d_array)
        


    def detect_2d_pose(self, rgb_img, vis_img, pred_classes, pred_boxes, pred_scores, pred_masks):
        boxes, scores, is_obj_ids, rgb_crops, is_masks = [], [], [], [], []
        poses_2d = []
        self.object_ids = [] 
        x_min, x_max, y_min, y_max =  self.params["roiofroi"]
        vis_img = cv2.putText(vis_img, "ROI", (int(x_min)-5, int(y_min)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 153, 0), 1, cv2.LINE_AA)
        vis_img = cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 153, 0), 2)
        for itr, (label, (x1, y1, x2, y2), score, mask) in enumerate(zip(pred_classes, pred_boxes, pred_scores, pred_masks)):
            if score < self.params["is_thresh"]:
                continue
            if x1 < x_min or x2 > x_max or y1 < y_min or y2 > y_max:
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
                angle, p1, vis_img, is_stable = identify_bracket(self.bracket_sock, rgb_crop, vis_img.copy(), mask.copy(), cntr, p1, p2)
                text = "stable" if is_stable else "unstable"
                if not is_stable: score = 0
<<<<<<< HEAD
                vis_img = cv2.putText(vis_img, text, (int(x1)-5, int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.idx2color[label], 1, cv2.LINE_AA)
=======
                vis_img = cv2.putText(vis_img, text, (int(x1)-5, int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.idx2color[label], 1, cv2.LINE_AA)
>>>>>>> ecfe62557128c14e9fc8982fc3e8999aa42a9885
            
            elif self.params["class_names"][label+1] == "ikea_stefan_bolt_side":
                angle, p1, vis_img = identify_side(self.side_sock, rgb_img.copy(), self.params["width"], self.params["height"], self.params["crop_offset"], vis_img.copy(), mask.copy(), cntr, p1, p2)
            
            elif self.params["class_names"][label+1] == "ikea_stefan_bolt_hip":
                angle, p1, vis_img = identify_hip(vis_img.copy(), mask.copy(), cntr, p1, p2)
            
            elif self.params["class_names"][label+1] == "ikea_stefan_pin":
                angle, p1, vis_img = identify_pin(vis_img, cntr, p1)
            
            object_id = self.params["class_names"][label+1] 
            self.object_ids.append(object_id)
            # visualize results 
            # + ' (' + str(np.rad2deg(angle))[:3] + ')'
            text = self.params["class_names"][label+1].split('_')[-1] + '(' + str(score)[:4] + ')'
            # vis_img = cv2.putText(vis_img, text, (int(x1)-5, int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.idx2color[label], 1, cv2.LINE_AA)
            vis_img = draw_axis(vis_img, cntr, p1, (127, 0, 255), 5)

            pose_2d = {"px": cntr[0], "py": cntr[1], "angle": angle, "score": score, "bbox": [x1, x2, y1, y2]}
            poses_2d.append(pose_2d)
        return poses_2d, vis_img

    def convert_2d_to_3d_pose(self, header, poses_2d, cloud_cam):
        
        self.object_poses = PoseArray()
        self.object_poses.header = header
        self.object_poses.poses = []
        self.grasp_poses = PoseArray()
        self.grasp_poses.header = header
        self.grasp_poses.poses = []
        for pose_2d in poses_2d:
            start_time = time.time()
            px = pose_2d["px"]
            py = pose_2d["py"]
            angle = pose_2d["angle"]
            # get points xyz around px, py
            is_mask = np.zeros([self.params["original_height"], self.params["original_width"]])
            is_mask[py-self.params["pix_near"]:py+self.params["pix_near"], px-self.params["pix_near"]:px+self.params["pix_near"]] = 1
            cloud_center = orh.crop_with_2dmask(copy.deepcopy(cloud_cam), is_mask, self.K_cropped)
            cloud_center_npy = np.asarray(cloud_center.points)
            mask = (np.nan_to_num(cloud_center_npy) != 0).any(axis=1)
            cloud_center_npy = cloud_center_npy[mask]
            pos = np.median(cloud_center_npy, axis=0)
            if (np.isnan(pos)).any():
                rospy.logwarn("NaN position. consider to increase the thresh")
                continue
            # get normal vector around px, py
            # normal_center_npy = np.asarray(cloud_center.normals)
            # normal = - np.median(normal_center_npy, axis=0)
            # rot = tf_trans.rotation_matrix

            H_cam2obj = np.eye(4)
            H_cam2obj[:3, 3] = pos
            H_cam2obj[:3, :3] = tf_trans.rotation_matrix(angle + np.deg2rad(self.params["angle_offset"]), (0, 0, 1))[:3, :3]
            H_map2obj = np.matmul(self.H_map2cam, H_cam2obj)
            pos = H_map2obj[:3, 3]
            quat = tf_trans.quaternion_from_matrix(H_map2obj)
            object_pose = orh.pq_to_pose(pos, quat)
            self.object_poses.poses.append(object_pose)
            self.grasp_poses.poses.append(object_pose)

        return self.object_poses

    def get_object_pose_array(self, msg):
        
        return [self.object_poses, self.grasp_poses, self.object_ids]



if __name__ == '__main__':

    kitting_manager = KittingManager()
    rospy.spin()



