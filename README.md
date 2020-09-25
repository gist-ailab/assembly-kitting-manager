# Assembly Kitting Manager

## Features
- Spatio-temporal median filter
- Currently, support only a single instance for each class
- Support classes: `"ikea_stefan_bottom", "ikea_stefan_long", "ikea_stefan_middle", "ikea_stefan_short", "ikea_stefan_side_left", "ikea_stefan_side_right"`

## To Do

- list up ros dependencies

## Getting Started

- [assembly_msgs](https://github.com/psh117/assembly_msgs)
- [assembly_camera_manager](https://github.com/SeungBack/assembly_camera_manager)
- [assembly_part_recognition](https://github.com/SeungBack/assembly_part_recognition)


## Published Topics and Servcie
#### `/assembly/detections/sptmpfilt`
- message type: `vision_msgs/Detection3DArray`
- Estimated 6D object pose after applying spatio-temporal median filter

#### `/assembly/markers/sptmpfilt`
- message type: `visualization_msgs/MarkerArray`
- Visualization of 6D object pose after applying spatio-temporal median filter

#### `/get_object_pose`
- message type: `assembly_msgs/GetObjectPose`
```
rosservice call /get_object_pose "target_object: 'ikea_stefan_bottom'"
```

#### `/get_object_pose_array`
- message type: `assembly_msgs/GetObjectPoseArray`
```
rosservice call /get_object_pose_array
```

## How to use
### Single Camera Setup (Kinect Azure)
1. launch k4a driver
```
$ ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch color_resolution:=720P depth_mode:=NFOV_2X2BINNED fps:=5  tf_prefix:=azure1_
```
2. launch k4a manager 
```
$ roslaunch assembly_camera_manager single_azure_manager.launch target_fiducial_id:="1"
```
3. camera to map calibration 
```
$ rosservice call /azure1/extrinsic_calibration "target_fiducial_ids: [1]"
```
4. 6d object pose estimation using MPAAE
```
$ roslaunch assembly_part_recognition single_azure_mpaae.launch yaml:=single_azure_mpaae_SNU
```
5. launch kitting manager
```
$ roslaunch assembly_kitting_manager kitting_manager.launch
```

## Authors
* **Seunghyeok Back** [seungback](https://github.com/SeungBack)

## License
This project is licensed under the MIT License

## Acknowledgments
This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.