import h5py  # 导入工具包
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

img_data_1 = []
img_data_2 = []
img_data_3 = []
img_data_4 = []
img_data_5 = []
img_label_1 = []
img_label_2 = []
img_label_3 = []

img_name = []
dataset_path = "E:/video_frame_train"
face_filedir = os.listdir(dataset_path)

order = 1
for i in range(60):
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
            img_name.append(file_name_1.encode())
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
            img_label_1.append(label_temp_2)
            img_label_2.append(label_temp_3)
            img_label_3.append(label_temp_4)
            temp_1 = cv2.resize(label_temp_1, (56, 56), interpolation=cv2.INTER_LINEAR)
            temp_2 = cv2.resize(label_temp_2, (28, 28), interpolation=cv2.INTER_AREA)
            temp_3 = cv2.resize(label_temp_3, (28, 28), interpolation=cv2.INTER_AREA)
            temp_4 = cv2.resize(label_temp_4, (28, 28), interpolation=cv2.INTER_AREA)
            temp_5 = cv2.resize(label_temp_5, (28, 28), interpolation=cv2.INTER_AREA)
            img_data_1.append(temp_1)
            img_data_2.append(temp_2)
            img_data_3.append(temp_3)
            img_data_4.append(temp_4)
            img_data_5.append(temp_5)
            # ax = plt.subplot("221")
            # ax.imshow(label_temp_1)
            # ax = plt.subplot("222")
            # ax.imshow(temp_1)
            # ax = plt.subplot("223")
            # ax.imshow(temp_2)
            # ax = plt.subplot("224")
            # ax.imshow(temp_3)
            # plt.show()
        else:
            order = 1
            print("{} {} {} not found".format(file_name_1, file_name_2, file_name_3))
            break

with h5py.File('video_train_x4.h5', 'w') as f:
    f.create_dataset('data_1', data=img_data_1)
    # f.create_dataset('data_2', data=img_data_2)
    # f.create_dataset('data_3', data=img_data_3)
    # f.create_dataset('data_4', data=img_data_4)
    # f.create_dataset('data_5', data=img_data_5)
    f.create_dataset('label_1', data=img_label_1)
    f.create_dataset('name', data=img_name)
    # f.create_dataset('label_2', data=img_label_2)
    # f.create_dataset('label_3', data=img_label_3)
    f.close()  # 关闭文件
print('h5 file Created! ')
