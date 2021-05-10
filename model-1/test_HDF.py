import numpy as np
import cv2
from torch.utils.data.dataloader import DataLoader
from getdata import DatasetFromHdf5_frame_1, DatasetFromHdf5_frame_3, DatasetFromHdf5_frame_5
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

landmark_path = "E:/feng_project/py_project/video_SR_1/video_1.txt"
picture_path = "E:/feng_project/video_1/"
file_landmark = open(landmark_path,'r')
landmark = []
name_index = []
for line in file_landmark:
    temp = list(line.strip('\n').split('\t'))
    name_index.append(temp[0])
    landmark.append(temp[1:])


batch_size = 4
index = np.zeros((4), dtype=np.int16)
train_set = DatasetFromHdf5_frame_3('E:/feng_project/py_project/video_SR_1/video_train_x8_frame_3.h5')
training_data_loader = DataLoader(dataset=train_set, num_workers=0,
                                  batch_size=4, shuffle=True)
mark_area = np.zeros((batch_size, 16), dtype=np.int16)
for iteration, batch in enumerate(training_data_loader, 1):
    input_1, input, input_2, target, name = Variable(batch[0]), Variable(batch[1], requires_grad=False), \
                          Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False),\
                          batch[4]
    input = torch.cat((input_1, input, input_2), 3)
    input = input.cuda()
    target = target.cuda()
    input = input.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)

    for i in range(len(name)):
        index[i] = name_index.index(name[i])
        print("{} is found in index {}".format(name[i], index[i]))
        print("landmark : {}".format(landmark[index[i]]))
        land_temp = list(map(int, landmark[index[i]][0:10]))
        img = cv2.imread(picture_path + name[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.rectangle(img, (land_temp[0] - 20, land_temp[1] + 20), (land_temp[0] + 20, land_temp[1] - 20),
                      (0, 0, 255), 2) ##左眼
        cv2.rectangle(img, (land_temp[2] - 20, land_temp[3] + 20), (land_temp[2] + 20, land_temp[3] - 20),
                      (0, 0, 255), 2) ##右眼
        cv2.rectangle(img, (land_temp[4] - 20, land_temp[5] + 20), (land_temp[4] + 20, land_temp[5] - 20),
                      (0, 0, 255), 2) ##鼻子
        cv2.rectangle(img, (land_temp[6] - 20, max(land_temp[7], land_temp[9]) + 20),
                      (land_temp[8] + 20, min(land_temp[7], land_temp[9]) - 20), (0, 0, 255), 2) ##嘴巴
        ax = plt.subplot("231")
        ax.imshow(img)
        mark_area[i] = [land_temp[0] - 20, land_temp[0] + 20, land_temp[1] - 20, land_temp[1] + 20,
                        land_temp[2] - 20, land_temp[2] + 20, land_temp[3] - 20, land_temp[3] + 20,
                        land_temp[4] - 20, land_temp[4] + 20, land_temp[5] - 20, land_temp[5] + 20,
                        land_temp[6] - 20, land_temp[7] + 20,
                        min(land_temp[7], land_temp[9]) - 20, max(land_temp[7], land_temp[9]) + 20]
        mark_area[mark_area < 0] = 0
        mark_area[mark_area > 223] = 223
        eye_left_target = target[i, :, mark_area[i][2]:mark_area[i][3], mark_area[i][0]:mark_area[i][1]]
        eye_left_target = np.uint8(eye_left_target.squeeze().cpu().numpy().transpose(1, 2, 0))
        eye_right_target = target[i, :, mark_area[i][6]:mark_area[i][7], mark_area[i][4]:mark_area[i][5]]
        eye_right_target = np.uint8(eye_right_target.squeeze().cpu().numpy().transpose(1, 2, 0))
        nose_target = target[i, :, mark_area[i][10]:mark_area[i][11], mark_area[i][8]:mark_area[i][9]]
        nose_target = np.uint8(nose_target.squeeze().cpu().numpy().transpose(1, 2, 0))
        mouth_target = target[i, :, mark_area[i][14]:mark_area[i][15], mark_area[i][12]:mark_area[i][13]]
        mouth_target = np.uint8(mouth_target.squeeze().cpu().numpy().transpose(1, 2, 0))
        ax = plt.subplot("232")
        ax.imshow(eye_left_target)
        ax = plt.subplot("233")
        ax.imshow(eye_right_target)
        ax = plt.subplot("234")
        ax.imshow(nose_target)
        ax = plt.subplot("235")
        ax.imshow(mouth_target)
        plt.show()
