# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys

#TODO : this is a temporary expedient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from centermask.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"

import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models

    #############################################################
    #추가한 코드 
    
    cfg.MODEL.WEIGHTS =  "/data_2/jongwon/output/stefan_dcn/model_0043999.pth"

    num_classes = 5
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes

    #############################################################

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        #default="configs/stefan/Base-CenterMask-VoVNet.yaml",
        default="configs/stefan/centermask_V_99_eSE_dcn_FPN_ms_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")[720*0:720*1,1280:1280*2]
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                path, len(predictions["instances"]), time.time() - start_time
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
            
        else:
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit
    
