import math
import numpy as np
import torch

def calc_psnr(img1, img2):
    temp = ((img1*255.0).int() - (img2*255.0).int()).float()
    return 20. * torch.log10(255. / torch.sqrt(torch.mean(temp ** 2)))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count