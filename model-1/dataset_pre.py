import os
from PIL import Image
import re
import cv2
import numpy as np

path = "E:/feng_project/video_dataset_train_GT"
save_path_x2 = "E:/feng_project/video_dataset_train_x2"
save_path_x4 = "E:/feng_project/video_dataset_train_x4"
scene = os.listdir(path)

# for j in range(len(scene)):
#     filedir_temp = path + '/' + scene[j]
#     if os.path.isdir(filedir_temp):
#         file_list = os.listdir(filedir_temp)
#         for i in range(len(file_list)):
#             img = Image.open(filedir_temp + '/' + file_list[i]).convert('RGB')
#             img = img.resize((1280, 720), Image.ANTIALIAS)
#             file_name = 'scene_' + str(j) + '_' + str(i) + '.jpg'
#             if (os.path.exists(save_path)):
#                 img.save(save_path + '/' + file_name, quality=100)
#             else:
#                 os.mkdir(save_path)
#                 img.save(save_path + '/' + file_name, quality=100)
#             print("{} process finished!".format(file_list[i]))

for j in range(len(scene)):
    file_list = path + '/' + scene[j]
    img = Image.open(file_list).convert('RGB')
    img_x2 = img.resize((1280 // 2, 720 // 2), Image.BICUBIC)
    img_x4 = img.resize((1280 // 4, 720 // 4), Image.BICUBIC)
    file_name_x2 = re.sub('.jpg', '_x2', scene[j]) + '.jpg'
    file_name_x4 = re.sub('.jpg', '_x4', scene[j]) + '.jpg'
    if (os.path.exists(save_path_x2)):
        img_x2.save(save_path_x2 + '/' + file_name_x2, quality=100)
    else:
        os.mkdir(save_path_x2)
        img_x2.save(save_path_x2 + '/' + file_name_x2, quality=100)
    if (os.path.exists(save_path_x4)):
        img_x4.save(save_path_x4 + '/' + file_name_x4, quality=100)
    else:
        os.mkdir(save_path_x4)
        img_x4.save(save_path_x4 + '/' + file_name_x4, quality=100)
    print("{} process finished!".format(file_list))