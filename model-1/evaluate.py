### this file aims to evaluate PSNR SSIM and FACE RECOGNITION
import torch
import pytorch_ssim
from getdata import DatasetFromHdf5_frame_1, DatasetFromHdf5_frame_3, DatasetFromHdf5_frame_5
from network import Video_SR_1
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from func import AverageMeter, calc_psnr
import face_recognition
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("===> Loading evaluation dataset")
eval_dataset = DatasetFromHdf5_frame_1("./eval_x4.h5")
eval_data_loader = DataLoader(dataset=eval_dataset, num_workers=0,
                              batch_size=1, shuffle=False)

print("===> Building video_sr_1 model")
model = Video_SR_1(4).cuda()
checkpoint = torch.load('./output/video_sr_gan2_x4/best.pth')
model.load_state_dict(checkpoint)

psnr_ori = 0
ssim_ori = 0
distance_ori = 0
psnr_sr = 0
ssim_sr = 0
distance_sr = 0
count_face = 0
count = 0
for iteration, batch in enumerate(eval_data_loader, 1):
    count = count + 1
    count_face = count_face + 1
    input_1, target, name = Variable(batch[0], requires_grad=False), \
                            Variable(batch[1], requires_grad=False), \
                            batch[2]
    input = input_1.permute(0, 3, 1, 2) / 255.0
    target = target.permute(0, 3, 1, 2) / 255.0

    input = input.cuda()
    target = target.cuda()
    with torch.no_grad():
        preds = model(input)
        preds = preds.clamp(0.0, 1.0)

        preds_t = preds.permute(0, 2, 3, 1)
        input_t = input.permute(0, 2, 3, 1)
        target_t = target.permute(0, 2, 3, 1)
        img_1 = input_t[0, :, :, :].detach().cpu().numpy()
        img_2 = preds_t[0, :, :, :].detach().cpu().numpy()
        img_3 = target_t[0, :, :, :].detach().cpu().numpy()
        plt.imshow(img_1)
        plt.imshow(img_2)
        plt.imshow(img_3)

        psnr_model = calc_psnr(preds, target)
        pic_ssim = pytorch_ssim.ssim(preds, target)

        input = torch.squeeze(input).cpu().numpy()
        input = np.transpose(input, (1, 2, 0)) * 255.0
        input_bic = np.array(Image.fromarray(input.astype('uint8')).convert('RGB').resize((144, 144))) / 255.0
        input_bic = torch.tensor(input_bic).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        input_bic = torch.tensor(input_bic, dtype=torch.float32).cuda()
        psnr_bic = calc_psnr(input_bic, target)
        bic_ssim = pytorch_ssim.ssim(input_bic, target)

        input_bic = torch.squeeze(input_bic).cpu().numpy()
        input_bic = np.transpose(input_bic, (1, 2, 0)) * 255.0
        input_bic = np.array(Image.fromarray(input_bic.astype('uint8')).convert('RGB').resize((112, 112)))

        preds = torch.squeeze(preds).cpu().numpy()
        preds = np.transpose(preds, (1, 2, 0)) * 255.0
        target = torch.squeeze(target).cpu().numpy()
        target = np.transpose(target, (1, 2, 0)) * 255.0
        preds = np.array(Image.fromarray(preds.astype('uint8')).convert('RGB').resize((112, 112)))
        target = np.array(Image.fromarray(target.astype('uint8')).convert('RGB').resize((112, 112)))
        size_bic = face_recognition.face_encodings(input_bic)
        size_sr = face_recognition.face_encodings(preds)
        if ((len(size_sr) == 0) | (len(size_bic) == 0)):
            count_face = count_face - 1
            distance = 0
            distance_bic = 0
        else:
            input_encode = face_recognition.face_encodings(input_bic)[0]
            preds_encode = face_recognition.face_encodings(preds)[0]
            target_encode = face_recognition.face_encodings(target)[0]
            distance = face_recognition.face_distance([target_encode], preds_encode)
            distance_bic = face_recognition.face_distance([target_encode], input_encode)

        psnr_ori = psnr_ori + psnr_bic
        ssim_ori = ssim_ori + bic_ssim
        distance_ori = distance_ori + distance_bic
        psnr_sr = psnr_sr + psnr_model
        ssim_sr = ssim_sr + pic_ssim
        distance_sr = distance_sr + distance
        print(count)
        # print('distance_sr:{}, bic:{}'.format(distance, distance_bic))
        # print('psnr_sr:{}, bic:{}'.format(psnr_model, psnr_bic))
        # print('ssim_sr:{}, bic:{}'.format(pic_ssim, bic_ssim))

print('ave_distance_sr:{}, bic:{}'.format(distance_sr / count_face, distance_ori / count_face))
print('ave_psnr_sr:{}, bic:{}'.format(psnr_sr / count, psnr_ori / count))
print('ave_ssim_sr:{}, bic:{}'.format(ssim_sr / count, ssim_ori / count))

params = list(model.parameters())#所有参数放在params里
k = 0
for i in params:
   l = 1
   for j in i.size():
        l*=j     #每层的参数存入l，这里也可以print 每层的参数
   k = k+l   #各层参数相加
print("all params:"+ str(k))   #输出总的参数