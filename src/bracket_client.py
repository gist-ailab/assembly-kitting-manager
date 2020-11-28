#!/usr/bin/python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from torchvision import datasets, transforms
import numpy as np
import itertools
import timm
import torchvision.transforms as T
from easy_tcp_python2_3 import socket_utils as su
import yaml
from pathlib import Path
import cv2
import time

if __name__ == "__main__" :

    yaml_path = os.path.join(Path(__file__).parent.parent, "params", "azure_centermask_GIST.yaml")
    with open(yaml_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params["bracket_gpu_id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(params["bracket_model_name"], pretrained=False)
    model.to(device)
    model.eval()
    checkpoint = torch.load(params["bracket_weight_path"])
    model.load_state_dict(checkpoint)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    sock = su.initialize_client(params["tcp_ip"], params["bracket_tcp_port"])

    while True:
        img = su.recvall_image(sock) 
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite('/home/demo/catkin_ws/src/assembly_kitting_manager/src/0/{}.png'.format(time.time()), img)
        # print(time.time())
        # time.sleep(0.3)
        img = transform(img).unsqueeze(0)
        output = model(img.to(device))

        topk=(1,)
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = np.bool(pred.t()[0].cpu().detach().numpy())
        pred = not pred
        su.sendall_pickle(sock, pred)

    sock.close()
