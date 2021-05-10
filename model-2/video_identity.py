from PIL import Image
from images2gif import writeGif
import os
import numpy as np
import imageio
import cv2

count = 1
outfilename = "demo-4-all.gif" # 转化的GIF图片名称
orignal = './data/test_video_set/sr-demo-4/video_clip/'
demo = './results/video-demo-4/'
identity = './Figs/demo-4/'

filenames_ori = 'clip'+str(count)+'.png'
filenames_dem = 'sr_'+str(count)+'.png'

frames = []
for count in range(2,119):
    filenames_ori = 'clip' + str(count) + '.png'
    filenames_dem = 'sr_' + str(count).rjust(2,'0') + '.png'
    filenames_pic = 'filename' + str(count) + '.png'

    im = Image.open(orignal + filenames_ori)             # 将图片打开，本文图片读取的结果是RGBA格式，如果直接读取的RGB则不需要下面那一步
    im = im.convert("RGB")                  # 通过convert将RGBA格式转化为RGB格式，以便后续处理
    im = np.array(im)                       # im还不是数组格式，通过此方法将im转化为数组
    h,w,_ = im.shape

    im_sr = Image.open(demo + filenames_dem)
    im_sr = im_sr.convert("RGB")
    im_sr = np.array(im_sr)

    im_pic = Image.open(identity + filenames_pic)
    im_pic = im_pic.convert("RGB")
    im_pic = np.array(im_pic)
    im_pic = cv2.resize(im_pic, (w, h), interpolation=cv2.INTER_AREA)

    im_gif = np.uint8(np.zeros((h,w*2,3)))
    mid = w//2
    quar = w//4
    im_gif[:,:mid,:] = im[:,quar: quar+mid, :]
    im_gif[:,mid+1: w, :] = im_sr[:,quar: quar+mid, :]
    im_gif[:,w: , :] = im_pic

    frames.append(im_gif)                       # 批量化
# writeGif(outfilename, frames, duration=0.1, subRectangles=False) # 生成GIF，其中durantion是延迟，这里是1ms
imageio.mimsave(outfilename, frames, 'GIF', duration=0.1) # 生成方式也差不多