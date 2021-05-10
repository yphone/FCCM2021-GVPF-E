from PIL import Image
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from network import Video_SR_1
from func import AverageMeter, calc_psnr
from torch.autograd import Variable
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

model = Video_SR_1(8)
model = model.cuda()
model.eval()
model.load_state_dict(torch.load("output/video_sr_1_x8/best.pth"))

dataset_path = "E:/video_1_test"
face_filedir = os.listdir(dataset_path)

psnr_bic = 0
psnr_our = 0
order = 1
for i in range(60, 65):
    while (True):
        file_name_1 = "Scene_" + str(i + 1) + '_' + str(order) + '.jpg'
        file_name_2 = "Scene_" + str(i + 1) + '_' + str(order + 1) + '.jpg'
        file_name_3 = "Scene_" + str(i + 1) + '_' + str(order + 2) + '.jpg'
        file_name_4 = "Scene_" + str(i + 1) + '_' + str(order + 3) + '.jpg'
        file_name_5 = "Scene_" + str(i + 1) + '_' + str(order + 4) + '.jpg'
        if((file_name_1 in face_filedir)&
                (file_name_2 in face_filedir)&
                (file_name_3 in face_filedir)&
                (file_name_4 in face_filedir)&
                (file_name_5 in face_filedir)):
            order = order + 1
            label_temp_1 = cv2.imread(dataset_path + '/' + file_name_1)
            label_temp_2 = cv2.imread(dataset_path + '/' + file_name_2)
            label_temp_3 = cv2.imread(dataset_path + '/' + file_name_3)
            label_temp_4 = cv2.imread(dataset_path + '/' + file_name_4)
            label_temp_5 = cv2.imread(dataset_path + '/' + file_name_5)
            label_temp_1 = cv2.cvtColor(label_temp_1, cv2.COLOR_BGR2RGB)
            label_temp_2 = cv2.cvtColor(label_temp_2, cv2.COLOR_BGR2RGB)
            label_temp_3 = cv2.cvtColor(label_temp_3, cv2.COLOR_BGR2RGB)
            label_temp_4 = cv2.cvtColor(label_temp_4, cv2.COLOR_BGR2RGB)
            label_temp_5 = cv2.cvtColor(label_temp_5, cv2.COLOR_BGR2RGB)
            temp_1 = cv2.resize(label_temp_1, (28, 28), interpolation=cv2.INTER_AREA)
            temp_2 = cv2.resize(label_temp_2, (28, 28), interpolation=cv2.INTER_AREA)
            temp_3 = cv2.resize(label_temp_3, (28, 28), interpolation=cv2.INTER_AREA)
            temp_4 = cv2.resize(label_temp_4, (28, 28), interpolation=cv2.INTER_AREA)
            temp_5 = cv2.resize(label_temp_5, (28, 28), interpolation=cv2.INTER_AREA)
            input = np.concatenate((temp_1, temp_2, temp_3), axis=2).transpose(2, 0, 1)
            input = (torch.tensor(input.astype(np.float32))).unsqueeze(0).to('cuda')
            out = model(input)
            out = out.squeeze().to('cpu').detach().numpy()
            out[out < 0] = 0
            out[out > 255.] = 255.
            out = np.uint8(out.transpose(1, 2, 0))
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            image_bicu = cv2.resize(temp_2, (224, 224), interpolation=cv2.INTER_CUBIC)
            image_bicu = cv2.cvtColor(image_bicu, cv2.COLOR_BGR2RGB)
            label_temp_2 = cv2.cvtColor(label_temp_2, cv2.COLOR_BGR2RGB)
            psnr_bic = PSNR(image_bicu, label_temp_2) + psnr_bic
            psnr_our = PSNR(out, label_temp_2) + psnr_our
            cv2.imshow('bicubic', image_bicu)
            cv2.imshow('video_sr', out)
            cv2.waitKey(50)
        else:
            psnr_bic = psnr_bic/(order - 1)
            psnr_our = psnr_our / (order - 1)
            print("psnr of bicubic is {}".format(psnr_bic))
            print("psnr of video_sr is {}".format(psnr_our))
            cv2.destroyAllWindows()  # 关闭窗口
            order = 1
            break
