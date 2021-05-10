import cv2
import os
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
from PIL import Image

save_path = 'E:/feng_project/video_pic'
pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                    r_model_path="./original_model/rnet_epoch.pt",
                                    o_model_path="./original_model/onet_epoch.pt",
                                    use_cuda=False)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

def video_process(video_path):
    cap = cv2.VideoCapture(video_path)  # 打开相机
    while(True):
        ret, frame = cap.read()  # 捕获一帧图像
        if ret:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()  # 关闭相机
    cv2.destroyAllWindows()  # 关闭窗口

def video_resize_rename(video_path, scene_num):
    cap = cv2.VideoCapture(video_path)
    state = 0
    num = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            bboxs, landmarks = mtcnn_detector.detect_face(frame)
            img_bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # vis_face(img_bg, bboxs, landmarks, 'file_list')
            img = Image.fromarray(img_bg.astype('uint8')).convert('RGB')
            if state == 0:
                w, h = img.size
                state = 1
                r = max(bboxs[0][2] - bboxs[0][0], bboxs[0][3] - bboxs[0][1])
            if (len(bboxs) == 0):
                print("no face detected!")
            else:
                center_x = (bboxs[0][0] + bboxs[0][2]) / 2
                center_y = (bboxs[0][1] + bboxs[0][3]) / 2
                left_x = center_x - r / 2
                left_y = center_y - r / 2
                right_x = center_x + r / 2
                right_y = center_y + r / 2
                if (left_x <= 0): left_x = 0
                if (left_y <= 0): left_y = 0
                if (right_x >= w):  right_x = w
                if (right_y >= h):  right_y = h
                img_crop = img.crop((left_x, left_y, right_x, right_y))
                img_crop.save(save_path + '/' + 'Scene'+ '_' + str(scene_num + 1) + '_' + str(num + 1) + '.jpg')
                num = num + 1
                if num > 200:
                    break
                print("finish!")
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


path = 'E:/feng_project/video_set'
file_list = os.listdir(path)
name = ['.mp4','.avi','.rmvb','.mkv']
for i in range(len(file_list)):
    file_temp = path + '/' + file_list[i]
    extension = os.path.splitext(file_list[i])[1]
    if extension in name:
        print("processing {}".format(file_list[i]))
        video_resize_rename(file_temp, i)
