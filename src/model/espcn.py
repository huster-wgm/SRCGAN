import functools
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ESPCN(nn.Module):
    """
    Superresolution using an efficient sub-pixel convolutional neural network
    Reference:
        CVPR2016: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
        code: https://github.com/pytorch/examples/tree/master/super_resolution
        PixelShuffle: https://blog.csdn.net/gdymind/article/details/82388068
    """

    def __init__(self,
                 in_ch=3,
                 ou_ch=3,
                 upscale_factor=2,
                 base_kernel=64):
        super(ESPCN, self).__init__()
        kernels = [int(x * base_kernel) for x in [1, 1, 1 / 2]]

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(
            in_ch, kernels[0], kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(
            kernels[0], kernels[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            kernels[1], kernels[2], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            kernels[2], base_kernel * upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv5 = nn.Conv2d(
            base_kernel, ou_ch, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return self.conv5(x)
