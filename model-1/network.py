import math
from torch import nn
import torch

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)
        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class _Residual_Block(nn.Module):
    def __init__(self, dim = 32):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        self.relu = nn.PReLU(dim)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False, groups=3)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output

class Video_SR_1(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=96, s=96, num_res=3):
        super(Video_SR_1, self).__init__()
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.sub_mean = MeanShift(rgb_mean, -1)
        self.first_part = nn.Sequential(nn.Conv2d(num_channels, s, kernel_size=3, padding=3//2), nn.PReLU(s))
        self.res_part = self.make_layer(_Residual_Block(s), num_res)
        self.second_part = nn.Sequential(
                                         nn.Conv2d(s, s * 4, kernel_size=3, padding=3 // 2),
                                         nn.PixelShuffle(2))
        self.last_part = nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=s * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=s, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        )
        # self.last_part = nn.ConvTranspose2d(s, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
        #                                      output_padding = scale_factor-1)
        # self.add_mean = MeanShift(rgb_mean, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.first_part(x)
        x = self.res_part(x)
        x = self.second_part(x)
        x = self.last_part(x)
        # x = self.add_mean(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))