## this code aims to compare original face and super-re face
import face_recognition
import cv2
import matplotlib.pyplot as plt
import numpy as np

def update(ref, current):
    alpha = 0.8 ## set the coefficient
    if(np.linalg.norm(ref - current, axis=0) < 0.6): ## if the same face
        ref_norm = np.linalg.norm(ref, axis=0)
        current = current / (np.linalg.norm(current, axis=0))
        ref = ref / ref_norm
        imp = alpha * ref + (1-alpha) * current
        imp = imp/ (np.linalg.norm(imp, axis=0))
        imp = imp * ref_norm
    else:
        imp = ref
    return imp

plt.ion()                  # 开启一个画图的窗口
title_demo = 'face_distance'
xlab = 'frame'
ylab = 'distance'
plt.title(title_demo,fontsize='large',fontweight='bold') #设置字体大小与格式
plt.xlabel(xlab)
plt.ylabel(ylab)
ax = []                 # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
ay_2 = []                    # 定义一个 y 轴的空列表用来接收动态的数据
ay_3 = []

demo_directory = './results/sr-demo-3/' ## sr_02 sr_118
count = 2
ref_face = cv2.imread(demo_directory + 'sr_' + str(count).rjust(2,'0') + '.png')
ref_face = cv2.resize(ref_face, (112, 112), interpolation=cv2.INTER_AREA)
ref_vector = face_recognition.face_encodings(ref_face)[0]
ref_update = ref_vector
original_directory = './data/test_video_set/sr-demo-3/lr_x4_BI/'

for i in range(2,119):
    plt.clf()  # 清除之前画的图
    plt.title(title_demo, fontsize='large', fontweight='bold')  # 设置字体大小与格式
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(linestyle='-.')  # 显示网格

    test_face = cv2.imread(demo_directory + 'sr_' + str(i).rjust(2,'0') + '.png')
    test_face = cv2.resize(test_face, (112, 112), interpolation=cv2.INTER_AREA)
    test_vector = face_recognition.face_encodings(test_face)[0]
    distance = face_recognition.face_distance([ref_vector], test_vector)

    ori_face = cv2.imread(original_directory + 'lr' + str(i) + '.png')
    ori_face = cv2.resize(ori_face, (112, 112), interpolation=cv2.INTER_AREA)
    ori_vector = face_recognition.face_encodings(ori_face)[0]
    distance_2 = face_recognition.face_distance([ref_vector], ori_vector)

    distance_3 = face_recognition.face_distance([ref_update], test_vector)
    ref_update = update(ref_update, test_vector)

    ax.append(i)
    ay.append(distance)
    ay_2.append(distance_2)
    ay_3.append(distance_3)
    plt.plot(ax, ay, 'go-', label='SR_face', linewidth=1)  # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.plot(ax, ay_2, 'ro-', label='original_face', linewidth=1)  # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.plot(ax, ay_3, 'bo-', label='SR-improved', linewidth=1)  # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.legend(loc='lower right')
    plt.savefig('Figs/demo-3/filename' + str(i) + '.png')
    plt.pause(0.2)  # 暂停0.1秒

