import torch
from network import Video_SR_1
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import os
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
from PIL import Image
import time
from func import AverageMeter, calc_psnr

pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                    r_model_path="./original_model/rnet_epoch.pt",
                                    o_model_path="./original_model/onet_epoch.pt",
                                    use_cuda=False)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=16)

model = Video_SR_1(8)
model = model.cuda()
model.eval()
model.load_state_dict(torch.load("output/video_sr_1_x8/best.pth"))

demo_video = "E:/video_set/7.avi"  # 49 50
scale = 8
order = 0
temp = np.zeros((3, 28, 28, 3))
cap = cv2.VideoCapture(demo_video)  # 打开视频

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_frame = cv2.VideoWriter('3_frame.avi', fourcc, 20.0, (1280, 720), True)

while(True):
    ret, frame_original = cap.read()  # 捕获一帧图
    if ret == 0:
        break
    w, h, _ = frame_original.shape
    frame_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame_original, (h//scale, w//scale), interpolation=cv2.INTER_AREA)
    # bicubic = cv2.resize(frame, (h, w), interpolation=cv2.INTER_CUBIC)
    bicubic = cv2.resize(frame, (h, w), interpolation=cv2.INTER_AREA)
    if order == 0:
        frame_GT = [frame_original, frame_original, frame_original]
        frame_bicu = [bicubic, bicubic, bicubic]
    else:
        frame_GT = [frame_GT[1], frame_GT[2], frame_original]
        frame_bicu = [frame_bicu[1], frame_bicu[2], bicubic]

    if ret:
        # t = time.time()
        if order == 0:
            order = order + 1
            print('Initial...')
            video_fifo = [frame, frame, frame]
            box,_ = mtcnn_detector.detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            box_fifo = [box, box, box]
        else:
            video_fifo = [video_fifo[1], video_fifo[2], frame]
            box, _ = mtcnn_detector.detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            box_fifo = [box_fifo[1], box_fifo[2], box]
            order = order + 1

            if (len(box_fifo[0]) == 0)|(len(box_fifo[1]) == 0)|(len(box_fifo[2]) == 0):
                print('The {} frame can not detect face'.format(order))
                frame_bicu[1] = cv2.cvtColor(frame_bicu[1], cv2.COLOR_BGR2RGB)
                cv2.imshow('bicubic', frame_bicu[1])
                cv2.imshow('video_sr', frame_bicu[1])
                cv2.waitKey(1)
            else:
                r = max(box_fifo[0][0][2] - box_fifo[0][0][0], box_fifo[0][0][3] - box_fifo[0][0][1],
                        box_fifo[1][0][2] - box_fifo[1][0][0], box_fifo[1][0][3] - box_fifo[1][0][1],
                        box_fifo[2][0][2] - box_fifo[2][0][0], box_fifo[2][0][3] - box_fifo[2][0][1])  # 取3帧里面最大的框
                for i in range(len(box_fifo)):
                    center_x = (box_fifo[i][0][0] + box_fifo[i][0][2]) / 2
                    center_y = (box_fifo[i][0][1] + box_fifo[i][0][3]) / 2
                    left_x = center_x - r / 2
                    left_y = center_y - r / 2
                    right_x = center_x + r / 2
                    right_y = center_y + r / 2
                    if (left_x <= 0): left_x = 0
                    if (left_y <= 0): left_y = 0
                    if (right_x >= w):  right_x = w
                    if (right_y >= h):  right_y = h
                    temp_face = video_fifo[i][int(left_y):int(right_y), int(left_x):int(right_x), :] #待调试
                    temp[i] = cv2.resize(temp_face, (28, 28), interpolation=cv2.INTER_CUBIC)
                input = np.concatenate((temp[0], temp[1], temp[2]), axis=2).transpose(2, 0, 1)

                similar = np.mean((temp[0] - temp[1]) ** 2)
                if (similar < 5):
                    print(order)
                    frame_bicu[1] = frame_bicu[0]
                else:
                    input = (torch.tensor(input.astype(np.float32))).unsqueeze(0).to('cuda')
                    out = model(input)
                    out = out.squeeze().to('cpu').detach().numpy()
                    out[out < 0] = 0
                    out[out > 255.] = 255.
                    out = np.uint8(out.transpose(1, 2, 0))
                    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                    frame_bicu[1] = cv2.cvtColor(frame_bicu[1], cv2.COLOR_BGR2RGB)
                    cv2.imshow('bicubic', frame_bicu[1])
                    center_x = (box_fifo[1][0][0] + box_fifo[1][0][2]) / 2
                    center_y = (box_fifo[1][0][1] + box_fifo[1][0][3]) / 2
                    left_x = center_x - r / 2
                    left_y = center_y - r / 2
                    right_x = center_x + r / 2
                    right_y = center_y + r / 2
                    if (left_x <= 0): left_x = 0
                    if (left_y <= 0): left_y = 0
                    if (right_x >= w):  right_x = w
                    if (right_y >= h):  right_y = h
                    out_cat = cv2.resize(out, (int(right_x)*scale - int(left_x)*scale,
                                              int(right_y)*scale - int(left_y)*scale),
                                        interpolation=cv2.INTER_AREA)
                    frame_bicu[1][int(left_y)*scale:int(right_y)*scale, int(left_x)*scale:int(right_x)*scale, :] = out_cat

                cv2.imshow('video_sr', frame_bicu[1])
                # print(time.time() - t)
                cv2.waitKey(1)
            out_frame.write(frame_bicu[1])

    else:
        break
out_frame.release()
cap.release()  # 关闭相机
cv2.destroyAllWindows()  # 关闭窗口