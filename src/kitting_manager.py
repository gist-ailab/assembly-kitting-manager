#!/usr/bin/env python
import rospy
import message_filters
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from assembly_msgs.srv import GetObjectPose, GetObjectPoseArray
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf.transformations as tf_trans


from queue import Queue
import numpy as np
import scipy
import glob
import open3d


class KittingManager():

    def __init__(self):
        # initalize node
        rospy.init_node('kitting_manager')
        rospy.loginfo("Starting kitting_manager.py")
        self.pe_class_names = ["ikea_stefan_bottom", "ikea_stefan_long", "ikea_stefan_middle", 
                        "ikea_stefan_short", "ikea_stefan_side_left", "ikea_stefan_side_right"]
        # assumption: hypothesis.id = class_id
        # !TODO: support multiple instances for each class
        self.hypothesis_ids = range(len(self.pe_class_names))

        # tf
        detection3d_array = rospy.wait_for_message('/assembly/detections/icp', Detection3DArray)
        detection_frame = detection3d_array.header.frame_id
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        is_sucess = False
        while not is_sucess:
            try:
                transform_cam_to_map = self.tf_buffer.lookup_transform("map", detection_frame, rospy.Time(), rospy.Duration(1.0)).transform
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)
                rospy.sleep(0.5)
            is_sucess = True
        # !TODO: camera_rgb_frame -> map 
        rospy.loginfo("Get TF from  map -> {}".format(detection_frame))
        self.H_cam_to_map = tf_trans.quaternion_matrix([
                transform_cam_to_map.rotation.x, transform_cam_to_map.rotation.y, 
                transform_cam_to_map.rotation.z, transform_cam_to_map.rotation.w])
        self.H_cam_to_map[:3, 3] = [
                    transform_cam_to_map.translation.x, 
                    transform_cam_to_map.translation.y, 
                    transform_cam_to_map.translation.z]

        self.params = rospy.get_param("kitting_manager")
        self.sptmpfilt_size = self.params["sptmpfilt_size"]
        self.reject_thresh = self.params["hypothesis_thresh"]

        self.hypothesis_que_array = [] 
        for i in self.hypothesis_ids:
            self.hypothesis_que_array.append(Queue(maxsize=self.sptmpfilt_size))
        # subscriber
        detections_sub = rospy.Subscriber('/assembly/detections/icp', Detection3DArray, self.callback)
        # publisher
        self.filt_detections_pub = rospy.Publisher('/assembly/detections/sptmpfilt', Detection3DArray, queue_size=1)
        self.filt_markers_pub = rospy.Publisher('/assembly/markers/sptmpfilt', MarkerArray, queue_size=1)

        # ply model for visualization
        self.dims = []
        self.ply_model_paths = glob.glob(self.params["ply_model_dir"] + '/*.ply')
        self.ply_model_paths.sort()
        for ply_model in self.ply_model_paths:
            cloud = open3d.io.read_point_cloud(ply_model)
            self.dims.append(cloud.get_max_bound())

        self.pose_srv = rospy.Service('/get_object_pose', GetObjectPose, self.return_object_pose)
        self.pose_array_srv = rospy.Service('/get_object_pose_array', GetObjectPoseArray, self.return_object_pose_array)


    def callback(self, detection_array):
        ## put hypothesis into queue
        for detection in detection_array.detections:
            hypothesis = detection.results[0]
            # reject the hypothesis with low score
            if hypothesis.score > self.reject_thresh:
                # print("{} rejected : {} > {}".format(self.pe_class_names[hypothesis.id], hypothesis.score, self.reject_thresh))
                continue
            if self.hypothesis_que_array[hypothesis.id].qsize() == self.sptmpfilt_size:
                self.hypothesis_que_array[hypothesis.id].get()
            self.hypothesis_que_array[hypothesis.id].put(hypothesis)
        det_header = Header()
        det_header.stamp = rospy.Time()
        det_header.frame_id = "map"
        self.filt_detection_array = Detection3DArray()
        self.filt_detection_array.header = det_header
        rospy.loginfo_once("Applying spatio-temporal median filter")
        # loop over class_id
        for class_id, hypothesis_que in enumerate(self.hypothesis_que_array):
            if hypothesis_que.qsize() == 0:
                continue
            # get all hypothesis from t-N:t
            hypothesis_array = list(hypothesis_que.queue)
            trans_array = []
            quat_array = []
            ## apply spatio temporal median filter 
            for hypothesis in hypothesis_array:
                pos = hypothesis.pose.pose.position
                ori = hypothesis.pose.pose.orientation
                trans_array.append([pos.x, pos.y, pos.z])
                quat_array.append([ori.x, ori.y, ori.z, ori.w])
            trans_median = np.median(np.asarray(trans_array), axis=0)
            quat_median = np.median(np.asarray(quat_array), axis=0)
            # publish marker
            # H_cam_to_obj -> H_map_to_obj
            H_cam_to_obj = np.eye(4)
            H_cam_to_obj[:3, 3] = tf_trans.translation_matrix((trans_median))[:3, 3]
            H_cam_to_obj[:3, :3] = tf_trans.quaternion_matrix((quat_median))[:3, :3]
            T_map_to_obj = np.matmul(self.H_cam_to_map, H_cam_to_obj) 
            R_map_to_obj = np.matmul(self.H_cam_to_map, H_cam_to_obj)
            translation = T_map_to_obj[:3, 3]
            rotation = tf_trans.quaternion_from_matrix(R_map_to_obj)
            # H to ros transformation            
            pose_msg = PoseStamped()
            pose_msg.header = det_header
            pose_msg.pose.position.x = translation[0] 
            pose_msg.pose.position.y = translation[1] 
            pose_msg.pose.position.z = translation[2] 
            pose_msg.pose.orientation.x = rotation[0]
            pose_msg.pose.orientation.y = rotation[1]
            pose_msg.pose.orientation.z = rotation[2]
            pose_msg.pose.orientation.w = rotation[3]
            filt_detection = Detection3D()
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = class_id
            hypothesis.pose.pose = pose_msg.pose
            filt_detection.results.append(hypothesis)
            filt_detection.bbox.center = pose_msg.pose
            filt_detection.bbox.size.x = self.dims[class_id][0] * 0.001 * 2
            filt_detection.bbox.size.y = self.dims[class_id][1] * 0.001 * 2
            filt_detection.bbox.size.z = self.dims[class_id][2] * 0.001 * 2
            self.filt_detection_array.detections.append(filt_detection)

        self.filt_detections_pub.publish(self.filt_detection_array)
        self.publish_markers(self.filt_markers_pub, self.filt_detection_array, [255, 13, 13])

    def return_object_pose(self, msg):

        target_id = self.pe_class_names.index(msg.target_object) 
        target_detection = None
        is_sucess = False
        for detection in self.filt_detection_array.detections:
            if target_id == detection.results[0].id:
                target_detection = detection
                is_sucess = True     
              
        return [target_detection, is_sucess]

    def return_object_pose_array(self, msg):
        return self.filt_detection_array

    def publish_markers(self, publisher, detections_array, color):
        # Delete all existing markers
        markers = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        publisher.publish(markers)
        # Object markers
        markers = MarkerArray()
        for i, det in enumerate(detections_array.detections):
            name = self.ply_model_paths[det.results[0].id].split('/')[-1][5:-4]
            # cube marker
            marker = Marker()
            marker.header = detections_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.3
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox.size
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = detections_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 1.0
            marker.id = i
            marker.ns = "texts"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.z = 0.07
            marker.text = '{} ({:.2f})'.format(name, det.results[0].score)
            markers.markers.append(marker)

            # mesh marker
            marker = Marker()
            marker.header = detections_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.9
            marker.ns = "meshes"
            marker.id = i
            marker.type = Marker.MESH_RESOURCE
            marker.scale.x = 0.001
            marker.scale.y = 0.001
            marker.scale.z = 0.001
            marker.mesh_resource = "file://" + self.ply_model_paths[det.results[0].id]
            marker.mesh_use_embedded_materials = True
            markers.markers.append(marker)
        publisher.publish(markers)




if __name__ == '__main__':

    kitting_manager = KittingManager()
    rospy.spin()