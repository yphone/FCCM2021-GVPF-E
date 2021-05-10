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

f = h5py.File('demo-4.h5','r')
index = f['data'][:] #numpy 119*3

count = 1
while(count < 118):
    clip_name = './data/test_video_set/sr-demo-4/video_clip/clip'+str(count+1)+'.png'
    sr_name = './results/sr-demo-4/sr_'+str(count+1).rjust(2,'0')+'.png'
    clip_concat = cv2.imread(clip_name)
    sr = cv2.imread(sr_name)
    c_x = index[count][0]
    c_y = index[count][1]
    x_left = c_x -72
    x_right = c_x + 72
    y_left = c_y - 72
    y_right = c_y + 72
    clip_concat[y_left:y_right, x_left:x_right, : ] = sr
    save_name = './results/video-demo-4/sr_'+str(count+1).rjust(2,'0')+'.png'
    cv2.imwrite(save_name, clip_concat)
    count = count + 1
    print(count)
f.close()


