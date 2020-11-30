#!/usr/bin/python
import cv2
import numpy as np
import os
import time 
import json
import argparse
import glob
import multiprocessing as mp
import tqdm
import sys
import yaml
from pathlib import Path
import torch

from easy_tcp_python2_3 import socket_utils as su
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
sys.path.append(os.path.join(Path(__file__).parent, "centermask2"))
from demo.predictor import VisualizationDemo
from centermask.config import get_cfg
from easy_tcp_python2_3 import socket_utils as su

def nine_to_3x3(nine):
    itr = 0
    for i in 0, 1, 2:
        for j in 0, 1, 2:
            if itr == nine:
                return i, j
            else:
                itr += 1
            

def setup_cfg(params):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(params["is_config_file"])
    cfg.merge_from_list([])
    # Set score_threshold for builtin models
    cfg.MODEL.WEIGHTS = params["is_weight_path"] 
    num_classes = len(params["class_names"])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = params["is_thresh"] 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params["is_thresh"] 
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = params["is_thresh"] 
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = params["is_thresh"] 
    cfg.freeze()
    return cfg


if __name__ == "__main__" :

    yaml_path = os.path.join(Path(__file__).parent.parent, "params", "azure_centermask_SNU.yaml")
    with open(yaml_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    cfg = setup_cfg(params)
    os.environ["CUDA_VISIBLE_DEVICES"] = params["is_gpu_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==> Loading CenterMask on", device, params["is_gpu_id"])
    demo = VisualizationDemo(cfg)
    sock = su.initialize_client(params["tcp_ip"], params["centermask_tcp_port"])
    
    while True:
        img = su.recvall_image(sock) 
        predictions, vis_output = demo.run_on_image(img)

        # w = params["width"] // params["grid_size"]
        # h = params["height"] // params["grid_size"]
        # img_grids = []
        # vis_img = np.zeros_like(img)
        # img_grids = []
        # for i in range(params["grid_size"]):
        #     for j in range(params["grid_size"]):
        #         img_grids.append(img[h*i:h*(i+1), w*j:w*(j+1), :])
        # predictions, visualized_outputs = demo.run_on_multiple_image(img_grids)
        # for itr, vis_output in enumerate(visualized_outputs):
        #     i, j = nine_to_3x3(itr)
        #     vis_img[h*i:h*(i+1), w*j:w*(j+1), :] = vis_output.get_image()[:, :, ::-1]

        
        pred_masks = predictions["instances"].pred_masks.cpu().detach().numpy() # (N, H, W),
        pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().detach().numpy() # (x1, y1, x2, y2)
        pred_scores = predictions["instances"].scores.cpu().detach().numpy()  # a vector of N confidence scores.
        pred_classes = predictions["instances"].pred_classes.cpu().detach().numpy() # [0, num_categories).
        vis_img = cv2.resize(vis_output.get_image()[:, :, ::-1], (params["width"], params["height"] ))
        su.sendall_image(sock, vis_img) 
        su.sendall_pickle(sock, pred_masks)
        su.sendall_pickle(sock, pred_boxes)
        su.sendall_pickle(sock, pred_classes)
        su.sendall_pickle(sock, pred_scores)

    s.close()
