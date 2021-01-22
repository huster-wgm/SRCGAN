import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class SRCNN(torch.nn.Module):
    """
    Superresolution using a Deep Convolutional Network
    Reference:
        ECCV2014: Learning a Deep Convolutional Network for Image Super-Resolution
    """

    def __init__(self,
                 in_ch=3,
                 ou_ch=3,
                 upscale_factor=2,
                 base_kernel=64):
        super(SRCNN, self).__init__()
        kernels = [int(x * base_kernel) for x in [1, 1 / 2]]
        self.up = upscale_factor
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(
            in_ch, kernels[0], kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(
            kernels[0], kernels[1], kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(
            kernels[1], ou_ch, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.up, mode="bilinear")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x
