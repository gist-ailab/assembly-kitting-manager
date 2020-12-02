from math import atan2, cos, sin, sqrt, pi
import cv2
import numpy as np
import os
import yaml
from pathlib import Path
from easy_tcp_python2_3 import socket_utils as su
import struct
import time 


yaml_path = str(Path(__file__).parent.parent) + "/params/azure_centermask_GIST.yaml"
with open(yaml_path) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
height = params["height"]
width = params["width"]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between_2d_vector(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if v2_u[0] < 0:
        angle += np.pi
    return angle

def angle_between_2d_vector(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if v2_u[0] < 0:
        angle = 2*np.pi - angle
    return angle



def get_2d_orientation(cntr, p1):
    """
    return 2d orientation w.r.t [0, -1]
    """
    axis_dir = np.asarray(p1) - np.asarray(cntr)
    angle = angle_between_2d_vector(np.array([0, -1]), axis_dir)
    return angle

def get_bbox_offset(x1, x2, y1, y2, offset, w, h):

    x1 = int(max(0, x1-offset))
    y1 = int(max(0, y1-offset))
    x2 = int(min(w-1, x2+offset))
    y2 = int(min(h-1, y2+offset))
    return x1, x2, y1, y2

def get_xy2box(box, offset, w, h ):

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    x1 = int(max(0, x1-offset))
    y1 = int(max(0, y1-offset))
    x2 = int(min(w-1, x2+offset))
    y2 = int(min(h-1, y2+offset))
    return x1, x2, y1, y2

def draw_axis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    return img 


def identify_bracket(sock, rgb_crop, vis_img, mask, cntr, p1, p2):
    # bracket => compare the number of pixels & binary stable classification
    mask1, mask2, cntr, p1, p2 = divide_mask(mask, cntr, p1, p2)
    n_pix_1 = np.sum(mask1)
    n_pix_2 = np.sum(mask2)
    if n_pix_1 < n_pix_2:
        p1 = cntr + 2*(cntr-p1)
    su.sendall_image(sock, rgb_crop)
    pred_result = su.recvall_pickle(sock) 
    angle = get_2d_orientation(cntr, p1)
    return angle, p1, vis_img, bool(pred_result)

def identify_side(sock, rgb_img, w, h, offset, vis_img, mask, cntr, p1, p2):
# def identify_side(vis_img, rgb_crop, mask, cntr, p1, p2):

    mask1, mask2, cntr, p1, p2 = divide_mask(mask, cntr, p1, p2)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect1 = cv2.minAreaRect(max(contours1, key = cv2.contourArea))
    rect2 = cv2.minAreaRect(max(contours2, key = cv2.contourArea))
    w1 = min(rect1[1][0], rect1[1][1])
    w2 = min(rect2[1][0], rect2[1][1])
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)
    offset = 0
    x1, x2, y1, y2 = get_xy2box(box1, offset, w, h)
    rgb1 = np.uint8(rgb_img[y1:y2, x1:x2].copy())
    su.sendall_image(sock, rgb1)
    is_bolt = su.recvall_pickle(sock)
    if is_bolt: 
        cv2.drawContours(vis_img, [box1], 0, (160, 128, 32), 1)
    else:
        p1 = cntr + 2*(cntr-p1)
        cv2.drawContours(vis_img, [box2], 0, (160, 128, 32), 1)

    # x1, x2, y1, y2 = get_xy2box(box2, offset, w, h)
    # rgb2 = np.uint8(rgb_img[y1:y2, x1:x2].copy())
    # time.sleep(0.3)
    # su.sendall_image(sock, rgb2)
    # pred_label2 = bool(su.recvall_pickle(sock))
    # prob2 = bool(su.recvall_pickle(sock))

    # if prob1 and prob2:
    #     if prob1 > prob2: 
    #         pred_label = pred_label1
    #     else:
    #         pred_label = pred_label2
    # else:
    #     pred_label = pred_label1

    # if pred_label:
    #     p1 = cntr + 2*(cntr-p1)    
    #     cv2.drawContours(vis_img, [box2], 0, (128, 160, 32), 2)
    # else:
    #     cv2.drawContours(vis_img, [box1], 0, (128, 160, 32), 2)

    angle = get_2d_orientation(cntr, p1)
    return angle, p1, vis_img

def identify_hip(vis_img, mask, cntr, p1, p2):
    # hip => compare the number of pixels
    mask1, mask2, cntr, p1, p2 = divide_mask(mask, cntr, p1, p2)
    n_pix_1 = np.sum(mask1)
    n_pix_2 = np.sum(mask2)
    if n_pix_1 < n_pix_2:
        p1 = cntr + 5*(cntr-p1)
    angle = get_2d_orientation(cntr, p1)
    return angle, p1, vis_img

def identify_pin(vis_img, cntr, p1):
    # pin => PCA direction
    angle = get_2d_orientation(cntr, p1)
    vis_img = draw_axis(vis_img, cntr, p1, (255, 0, 0), 5)
    return angle, p1, vis_img

def divide_mask(mask, cntr, p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    cntr = np.asarray(cntr)
    p1_cntr = p1 - cntr
    p2_cntr = p2 - cntr
    crop_mask = np.zeros((height, width), np.uint8)
    pts = np.array([50*p2_cntr+cntr, 50*(p1_cntr+p2_cntr)+cntr, 50*(p1_cntr-p2_cntr)+cntr, -50*p2_cntr+cntr])
    _ = cv2.drawContours(crop_mask, np.int32([pts]), 0, 255, -1)
    mask1 = mask.copy()
    mask2 = mask.copy()
    mask1[crop_mask==0] = 0
    mask2[crop_mask>0] = 0
    return mask1, mask2, cntr, p1, p2

def pca_analysis(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    cv2.circle(img, cntr, 3, (0, 0, 255), 6)
    p1 = (cntr[0] + 0.01 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.01 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.01 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.01 * eigenvectors[1,1] * eigenvalues[1,0])
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return img, cntr, p1, p2, angle


def convert_2d_to_3d_position(cloud, u, v):
    width = cloud.width
    height = cloud.height
    point_step = cloud.point_step
    row_step = cloud.row_step
    array_pos = v*row_step + u*point_step
    bytesX = [ord(x) for x in cloud.data[array_pos:array_pos+4]]
    bytesY = [ord(x) for x in cloud.data[array_pos+4: array_pos+8]]
    bytesZ = [ord(x) for x in cloud.data[array_pos+8:array_pos+12]]
    byte_format=struct.pack('4B', *bytesX)
    X = struct.unpack('f', byte_format)[0]
    byte_format=struct.pack('4B', *bytesY)
    Y = struct.unpack('f', byte_format)[0]
    byte_format=struct.pack('4B', *bytesZ)
    Z = struct.unpack('f', byte_format)[0]
    return X, Y, Z