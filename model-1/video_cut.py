import cv2
import os
import numpy as np

bicu_video = 'bicu_frame.avi'
bicu = cv2.VideoCapture(bicu_video)  # 打开相机

GT_video = 'GT_frame.avi'
GT = cv2.VideoCapture(GT_video)  # 打开相机

frame1_video = '1_frame.avi'
frame_1 = cv2.VideoCapture(frame1_video)  # 打开相机

frame3_video = '3_frame.avi'
frame_3 = cv2.VideoCapture(frame3_video)  # 打开相机

frame5_video = '5_frame.avi'
frame_5 = cv2.VideoCapture(frame5_video)  # 打开相机

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_frame = cv2.VideoWriter('Result.avi', fourcc, 20.0, (1800, 720), True)

order = 0
while(True):
    ret_1, frame1 = bicu.read()  # 捕获一帧图像
    ret_2, frame2 = GT.read()  # 捕获一帧图像
    ret_3, frame3 = frame_1.read()  # 捕获一帧图像
    ret_4, frame4 = frame_3.read()  # 捕获一帧图像
    ret_5, frame5 = frame_5.read()  # 捕获一帧图像
    frame = np.zeros((720, 1800))
    if ret_1 & ret_2 & ret_3 & ret_4 & ret_5:
        if order < 400:
            if order == 0:
                text = 'We only use Face-SR to face in video'
                cv2.putText(frame, text, (400, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(3000)

                frame = np.zeros((720, 1800))
                text = 'Using 1 frame'
                cv2.putText(frame, text, (600, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(2000)
            else:
                frame = np.concatenate((frame1[:, 300:900, :], frame3[:, 300:900, :], frame2[:, 300:900, :]), axis=1)
                text_1 = "Bicubic"
                cv2.putText(frame, text_1, (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                text_2 = "Using 1 frame"
                cv2.putText(frame, text_2, (610, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                text_3 = "Ground-truth"
                cv2.putText(frame, text_3, (1210, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(20)
        elif order < 800:
            if order == 400:
                frame = np.zeros((720, 1800))
                text = 'Using 3 frame'
                cv2.putText(frame, text, (600, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(2000)
            else:
                frame = np.concatenate((frame1[:, 300:900, :], frame4[:, 300:900, :], frame2[:, 300:900, :]), axis=1)
                text_1 = "Bicubic"
                cv2.putText(frame, text_1, (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                text_2 = "Using 3 frame"
                cv2.putText(frame, text_2, (610, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                text_3 = "Ground-truth"
                cv2.putText(frame, text_3, (1210, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(20)
        elif order < 1200:
            if order == 800:
                text = 'Using 5 frame'
                frame = np.zeros((720, 1800))
                cv2.putText(frame, text, (600, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(2000)
            else:
                frame = np.concatenate((frame1[:, 300:900, :], frame5[:, 300:900, :], frame2[:, 300:900, :]), axis=1)
                text_1 = "Bicubic"
                cv2.putText(frame, text_1, (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                text_2 = "Using 5 frame"
                cv2.putText(frame, text_2, (610, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                text_3 = "Ground-truth"
                cv2.putText(frame, text_3, (1210, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(20)
        else :
            if order == 1200:
                text = 'Using 1 frame and 3 frame and 5 frame'
                frame = np.zeros((720, 1800))
                cv2.putText(frame, text, (400, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(2000)
            else:
                frame = np.concatenate((frame3[:, 300:900, :], frame4[:, 300:900, :], frame5[:, 300:900, :]), axis=1)
                text_1 = "Using 1 frame"
                cv2.putText(frame, text_1, (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                text_2 = "Using 3 frame"
                cv2.putText(frame, text_2, (610, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                text_3 = "Using 5 frame"
                cv2.putText(frame, text_3, (1210, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (100, 200, 200), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(20)
        order = order + 1
        out_frame.write(frame)
    else:
        print(order)
        break

bicu.release()  # 关闭相机
GT.release()
frame_1.release()
frame_3.release()
frame_5.release()
out_frame.release()
cv2.destroyAllWindows()  # 关闭窗口