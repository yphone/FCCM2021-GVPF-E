import cv2
import os
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
from PIL import Image

file_path = 'E:/feng_project/video_pic'
save_path = 'E:/feng_project/video_1'
pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                    r_model_path="./original_model/rnet_epoch.pt",
                                    o_model_path="./original_model/onet_epoch.pt",
                                    use_cuda=False)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

file = os.listdir(file_path)
order = 0
for i in range(65):
    while(True):

        file_name = "Scene_" + str(i+1) + '_' + str(order + 1) + '.jpg'
        if file_name in file:
            img = cv2.imread(file_path + '/' + file_name)
            img_resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            bboxs, landmarks = mtcnn_detector.detect_face(img_resize)

            img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')

            f = open('video_1.txt', 'a')
            if (len(landmarks) != 0):
                order = order + 1
                img.save(save_path + '/' + file_name)
                f.write(file_name + '\t')
                for j in range(10):
                    f.write(str(int(landmarks[0][j])) + '\t')
                f.write('\n')
                f.close()
            else:
                order = 0
                f.close()
                print("Scene_{} finish".format(i + 1))
                break
        else:
            order = 0
            print("Scene_{} finish".format(i+1))
            break

