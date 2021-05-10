import numpy as np
from torch import nn
import torch

def face_loss(name, preds, target, landmark, name_index):

    batch_size = len(name)
    criterion = nn.MSELoss(size_average=False)
    index = np.zeros((batch_size), dtype=np.int16)
    mark_area = np.zeros((batch_size, 16), dtype=np.int16)

    for i in range(len(name)):
        index[i] = name_index.index(name[i])
        # print("{} is found in index {}".format(name[i], index[i]))
        # print("landmark : {}".format(landmark[index[i]]))
        land_temp = list(map(int, landmark[index[i]][0:10]))
        mark_area[i] = [land_temp[0] - 20, land_temp[0] + 20, land_temp[1] - 20, land_temp[1] + 20,
                        land_temp[2] - 20, land_temp[2] + 20, land_temp[3] - 20, land_temp[3] + 20,
                        land_temp[4] - 20, land_temp[4] + 20, land_temp[5] - 20, land_temp[5] + 20,
                        land_temp[6] - 20, land_temp[7] + 20,
                        min(land_temp[7], land_temp[9]) - 20, max(land_temp[7], land_temp[9]) + 20]
        mark_area[mark_area < 0] = 0
        mark_area[mark_area > 223] = 223
        eye_left_target = target[i, :, mark_area[i][2]:mark_area[i][3], mark_area[i][0]:mark_area[i][1]]
        eye_left_preds = preds[i, :, mark_area[i][2]:mark_area[i][3], mark_area[i][0]:mark_area[i][1]]
        eye_right_target = target[i, :, mark_area[i][6]:mark_area[i][7], mark_area[i][4]:mark_area[i][5]]
        eye_right_preds = preds[i, :, mark_area[i][6]:mark_area[i][7], mark_area[i][4]:mark_area[i][5]]
        nose_target = target[i, :, mark_area[i][10]:mark_area[i][11], mark_area[i][8]:mark_area[i][9]]
        nose_preds = preds[i, :, mark_area[i][10]:mark_area[i][11], mark_area[i][8]:mark_area[i][9]]
        mouth_target = target[i, :, mark_area[i][14]:mark_area[i][15], mark_area[i][12]:mark_area[i][13]]
        mouth_preds = preds[i, :, mark_area[i][14]:mark_area[i][15], mark_area[i][12]:mark_area[i][13]]
    left_eye_loss = criterion(eye_left_preds, eye_left_target)
    right_eye_loss = criterion(eye_right_preds, eye_right_target)
    nose_loss = criterion(nose_preds, nose_target)
    mouth_loss = criterion(mouth_preds, mouth_target)
    return left_eye_loss, right_eye_loss, nose_loss, mouth_loss
