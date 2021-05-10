import argparse
import os
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import random
from getdata import DatasetFromHdf5_frame_1, DatasetFromHdf5_frame_3, DatasetFromHdf5_frame_5
from network import Video_SR_1, Discriminator
from func import AverageMeter, calc_psnr
from torch.autograd import Variable
from face_loss import face_loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from loss import GeneratorLoss_1
from backbone.model_irse import IR_50
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, default='output/')
    parser.add_argument("--weights", default='./output/video_sr_only_face/best.pth', type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument("--cuda", default=True, help="use cuda?")
    parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchSize', type=int, default=48)
    parser.add_argument('--num-epochs', type=int, default=180)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'video_sr_gan2_x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    writer = SummaryWriter('runs/')

    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print("===> Loading training dataset")
    train_set = DatasetFromHdf5_frame_1("./video_train_x4.h5")
    # train_set = DatasetFromHdf5("fsrcnn_face_train.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads,
                                      batch_size=args.batchSize,shuffle=True)

    # print("===> Loading landmark and name")
    # landmark_path = "video_1.txt"
    # file_landmark = open(landmark_path, 'r')
    # landmark = []
    # name_index = []
    # for line in file_landmark:
    #     temp = list(line.strip('\n').split('\t'))
    #     name_index.append(temp[0])
    #     landmark.append(temp[1:])

    print("===> Loading evaluation dataset")
    eval_dataset = DatasetFromHdf5_frame_1("./video_test_x4.h5")
    eval_data_loader = DataLoader(dataset=eval_dataset, num_workers=args.threads,
                                      batch_size= args.batchSize, shuffle=True)

    print("===> Building video_sr_1 model")
    model = Video_SR_1(args.scale)
    # criterion = GeneratorLoss_1()
    criterion_2 = nn.MSELoss()

    netD = Discriminator()
    print('===> discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        netD = netD.cuda()
        # criterion = criterion.cuda()
        criterion_2 = criterion_2.cuda()

    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.weights))

    # print("===> Setting Optimizer")
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    BACKBONE_RESUME_ROOT = './ms1m-ir50/backbone_ir50_ms1m_epoch120.pth'
    BACKBONE = IR_50([112,112])
    print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
    BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    BACKBONE = BACKBONE.cuda()

    print("===> Training")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(args.start_epoch, args.num_epochs + 1):

        learn_rate = args.lr/(2 ** (epoch//40))
        # learn_rate = args.lr
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learn_rate)
        optimizerD = optim.Adam(netD.parameters())
        print("learning rate is {:6f}".format(learn_rate))

        for param_group in optimizer.param_groups:
            param_group["lr"] = learn_rate

        model.train()
        # netD.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_set) - len(train_set) % args.batchSize), ncols=150) as t:
            for iteration, batch in enumerate(training_data_loader, 1):
                input_1, target, name = Variable(batch[0], requires_grad=False), \
                                        Variable(batch[1], requires_grad=False), \
                                        batch[2]
                # img_1 = input_1[0, :, :, :].detach().cpu().numpy()
                # img_2 = target[0, :, :, :].detach().cpu().numpy()

                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.imshow(img_1 / 255)
                # plt.subplot(1, 2, 2)
                # plt.imshow(img_2 / 255)
                # plt.show()

                # input_1, target, name = Variable(batch[0], requires_grad=False), \
                #                                           Variable(batch[1], requires_grad=False), \
                #                                           batch[2]
                # input = input_1
                input = input_1.permute(0, 3, 1, 2)/255.0
                target = target.permute(0, 3, 1, 2)/255.0
                if args.cuda:
                    input = input.cuda()
                    target = target.cuda()

                # ### update D network
                # out = model(input)
                # netD.zero_grad()
                # fake_out = netD(out).mean()
                # real_out = netD(out).mean()
                # d_loss = torch.exp((fake_out**2))+ torch.exp((1-real_out)**2)
                # d_loss.backward(retain_graph=True)
                # # d_loss.backward()
                # optimizerD.step()
                #
                # ### update G network
                # model.zero_grad()
                # pic_loss, vgg_loss = criterion(out, target)
                # face_identity = criterion_2(out[:, :, 16:128, 16:128], target[:, :, 16:128, 16:128])
                # if epoch < 1:
                #     loss = pic_loss + face_identity / 2
                # else:
                #     fake_out = netD(out).mean()
                #     d_loss = torch.exp((1 - fake_out)**2)
                #     loss = pic_loss + face_identity / 2 + d_loss/100
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()


                ## v1.0 version expect gan-loss
                out = model(input)
                # pic_loss,vgg_loss = criterion(out, target)
                pic_loss = criterion_2(out, target)
                face_emphasize = criterion_2(out[:,:,16:128,16:128], target[:,:,16:128,16:128])
                face_identity = criterion_2(BACKBONE(out[:,:,16:128,16:128]),BACKBONE(target[:,:,16:128,16:128]))
                loss = pic_loss + face_emphasize/2 + face_identity/200000
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                img_1 = out[0, :, :, :].detach().cpu().numpy().transpose(1,2,0)
                img_2 = target[0, :, :, :].detach().cpu().numpy().transpose(1,2,0)
                #
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(img_1 )
                plt.subplot(1, 2, 2)
                plt.imshow(img_2 )
                plt.show()

                epoch_losses.update(loss.item(), len(input))
                t.set_postfix(loss='{:.6f}'.format(loss),
                              pic = '{:.6f}'.format(pic_loss),
                              emphasize = '{:.6f}'.format(face_emphasize/2),
                              identity='{:.6f}'.format(face_identity/200000)
                              # real_out='{:.6f}'.format(real_out),
                              # fake_out='{:.6f}'.format(fake_out))
                              )
                t.update(len(input))
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs))
        if epoch % 20 == 0:            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()
        eval_losses = AverageMeter()
        for iteration, batch in enumerate(eval_data_loader, 1):
            input_1, target, name = Variable(batch[0], requires_grad=False), \
                                    Variable(batch[1], requires_grad=False), \
                                    batch[2]
            input = input_1.permute(0, 3, 1, 2)/255.0
            target = target.permute(0, 3, 1, 2)/255.0
            if args.cuda:
                input = input.cuda()
                target = target.cuda()
            with torch.no_grad():
                preds = model(input)
                preds = preds.clamp(0.0, 1.0)
                # pic_loss,vgg_loss = criterion(preds, target)
                pic_loss = criterion_2(preds, target)
                # loss = pic_loss + vgg_loss
                loss = pic_loss
            # img_1 = preds[0, :, :, :].detach().cpu().numpy().transpose(1,2,0)
            # img_2 = target[0, :, :, :].detach().cpu().numpy().transpose(1,2,0)
            # #
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(img_1 )
            # plt.subplot(1, 2, 2)
            # plt.imshow(img_2 )
            # plt.show()
            # t = calc_psnr(preds, target)
            epoch_psnr.update(calc_psnr(preds, target), len(input))
            eval_losses.update(loss.item(), len(input))
        print('eval psnr: {:.6f}'.format(epoch_psnr.avg))
        print('eval loss: {:.6f}'.format(eval_losses.avg/args.batchSize))
        print('pic loss: {:.6f},face_loss: {:.6f}'.format(pic_loss/args.batchSize, face_identity/args.batchSize))
        writer.add_scalars('/x4_0', {'train_loss': epoch_losses.avg/args.batchSize,
                                           'eval_psnr_loss': epoch_psnr.avg,
                                           'eval_loss': eval_losses.avg/args.batchSize}, epoch)
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    writer.close()
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))