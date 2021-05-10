import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import os
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
from PIL import Image
import time
from modules import SOFVSR
import h5py

pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                    r_model_path="./original_model/rnet_epoch.pt",
                                    o_model_path="./original_model/onet_epoch.pt",
                                    use_cuda=False)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=16)

demo_video = './65.avi'  # 7 59 63 65
cap = cv2.VideoCapture(demo_video)  # 打开视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')

size_o_w = 1280
size_o_h = 720
size_i_w = 1280
size_i_h = 720
count = 1
box_list = []

while(count < 120):
    print(count)
    ret, frame_original = cap.read()  # 捕获一帧图
    if ret == 0:
        break

    if(count == 1):
        w, h, _ = frame_original.shape
        # box, _ = mtcnn_detector.detect_face(cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB))
        box, _ = mtcnn_detector.detect_face(frame_original)

        r = ((box[0][2] - box[0][0])+( box[0][3] - box[0][1]))/2
        center_x = (box[0][0] + box[0][2]) / 2
        center_y = (box[0][1] + box[0][3]) / 2
        left_x = np.int(center_x - r / 2)
        left_y = np.int(center_y - r / 2)
        right_x = np.int(center_x + r / 2)
        right_y = np.int(center_y + r / 2)
        if (left_x <= 0): left_x = 0
        if (left_y <= 0): left_y = 0
        if (right_x >= h):  right_x = h
        if (right_y >= w):  right_y = w

        size_i_w = np.int(36 * w // r)
        size_i_h = np.int(36 * h // r)
        size_o_w = np.int(144 * w // r)
        size_o_h = np.int(144 * h // r)

        frame = cv2.resize(frame_original, (size_i_h, size_i_w), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (size_o_h, size_o_w), interpolation=cv2.INTER_AREA)

        filename = './data/test_video_set/sr-demo-4/video_clip/clip' + str(count-1) + '.png'
        cv2.imwrite(filename, frame)

        file_face = './data/test_video_set/sr-demo-4/lr_x4_BI/lr'+str(count-1)+'.png'
        c_x = np.int(center_x * size_o_h / h)
        c_y = np.int(center_y * size_o_w / w)
        cv2.imwrite(file_face, cv2.resize(frame_original[left_y:right_y, left_x:right_x,:], (36,36), interpolation=cv2.INTER_AREA))
        box_list.append([c_x,c_y,count])

        count = count + 1

    else:
        # box, _ = mtcnn_detector.detect_face(cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB))
        box, _ = mtcnn_detector.detect_face(frame_original)

        # r = max(box[0][2] - box[0][0], box[0][3] - box[0][1])
        center_x = (box[0][0] + box[0][2]) / 2
        center_y = (box[0][1] + box[0][3]) / 2
        left_x = np.int(center_x - r / 2)
        left_y = np.int(center_y - r / 2)
        right_x = np.int(center_x + r / 2)
        right_y = np.int(center_y + r / 2)
        if (left_x <= 0): left_x = 0
        if (left_y <= 0): left_y = 0
        if (right_x >= h):  right_x = h
        if (right_y >= w):  right_y = w

        frame = cv2.resize(frame_original, (size_i_h, size_i_w), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (size_o_h, size_o_w), interpolation=cv2.INTER_AREA)
        filename = './data/test_video_set/sr-demo-4/video_clip/clip' + str(count-1) + '.png'
        cv2.imwrite(filename, frame)

        file_face = './data/test_video_set/sr-demo-4/lr_x4_BI/lr'+str(count-1)+'.png'
        c_x = np.int(center_x * size_o_h / h)
        c_y = np.int(center_y * size_o_w / w)
        cv2.imwrite(file_face, cv2.resize(frame_original[left_y:right_y, left_x:right_x,:], (36,36), interpolation=cv2.INTER_AREA))
        box_list.append([c_x,c_y,count])
        count = count + 1

cap.release()  # 关闭相机
with h5py.File('demo-4.h5', 'w') as f:
    f.create_dataset('data', data=box_list)
    f.close()  # 关闭文件
print('h5 file Created! ')
