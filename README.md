# Assembly Kitting Manager

## Features
- connector detection (CenterMask 2)
- Axis detection (PCA + Binary classification)
- Convert 2D pose to 3D pose

## To Do

- list up ros dependencies

## Getting Started

- [assembly_msgs](https://github.com/psh117/assembly_msgs)
- [assembly_camera_manager](https://github.com/SeungBack/assembly_camera_manager)



## Published Topics and Servcie

#### `/assembly/vis_kitting`
- message type: `sensor_msgs/Image`
- Visualized result in 2D image

#### `/assembly/connector_pose`
- message type: `geometry_msgs/PoseArray`
- Visualized result in 3D point cloud

#### `/get_object_pose_array`
- service type: `assembly_msgs/GetObjectPoseArray`
- return object poses, grasp_poses and object ids 
- 0: bracket, 1: bolt_side, 2: bolt_hip, 3: pin
```
rosservice call /get_object_pose_array
```


## How to use

### Recognition
1. Launch camera node and manager
```
$ ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch color_resolution:=1440P depth_mode:=NFOV_2X2BINNED fps:=5 tf_prefix:=azure1_
$ ass && roslaunch assembly_camera_manager single_azure_manager.launch 
```

2. Set camera pose from yaml
```
# gist
$ rosservice call /azure1/set_camera_pose "json_file: 'base_to_azure1_rgb_camera_link_20201119-133337'"
# snu
$ rosservice call /azure1/set_camera_pose "json_file: 'base_to_azure1_rgb_camera_link_20201123-160941'"
```
3. Launch kitting manager
```
$ ass && roslaunch assembly_kitting_manager kitting_manager.launch yaml:=azure_centermask_GIST
```

4. Instance segmentation using centermask
```
$ ass37 && python ~/catkin_ws/src/assembly_kitting_manager/src/centermask_client.py
```

5. Bracket
```
$ ass37 && python ~/catkin_ws/src/assembly_kitting_manager/src/bracket_client.py
```

6. Side
```
$ ass37 && python ~/catkin_ws/src/assembly_kitting_manager/src/side_client.py
```


## Authors
* **Seunghyeok Back** [seungback](https://github.com/SeungBack)

## License
This project is licensed under the MIT License

## Acknowledgments
This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.