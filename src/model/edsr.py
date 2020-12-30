import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def get_deconv_params(upscale_factor):
    """
    args:
        ratio:(str) upsample level
        H_out =(H_in−1)×stride[0]−2×pad[0]+ksize[0]+output_padding[0] 
    """
    if upscale_factor==2:
        kernel_size = 2
        stride=2
    elif upscale_factor==4:
        kernel_size = 2
        stride=4
    elif upscale_factor==8:
        kernel_size = 4
        stride=8
    output_padding = (stride-kernel_size)
    return kernel_size, stride, output_padding


def deconv(in_planes, out_planes, upscale_factor=2):
    """2d deconv"""
    kernel_size, stride, opad = get_deconv_params(upscale_factor)
    # print("DECONV", kernel_size, stride, opad)
    return nn.ConvTranspose2d(in_planes, out_planes, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=0, 
                              bias=False,
                              output_padding=opad,
                              groups=1)


class ResnetBlock(nn.Module):
    def __init__(self, num_channel, kernel=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.gn = nn.GroupNorm(32, num_channel) # replace BN => GN
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.gn(self.conv1(x))
        x = self.activation(x)
        x = self.gn(self.conv2(x))
        x = torch.add(x, residual)
        return x


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor, kernel=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2, kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel, stride, padding)

    def forward(self, x):
        x = self.ps(self.conv1(x))
        return x


class EDSR(nn.Module):
    """
    https://github.com/icpm/super-resolution/edit/master/EDSR/model.py
    """
    def __init__(self, in_ch, ou_ch, upscale_factor=2, base_channel=64, num_residuals=50):
        super(EDSR, self).__init__()

        self.input_conv = nn.Conv2d(in_ch, base_channel, kernel_size=3, stride=1, padding=1)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_channel, kernel=3, stride=1, padding=1))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)

        upscale = []
        for _ in range(int(math.log2(upscale_factor))):
            upscale.append(deconv(base_channel, base_channel, upscale_factor=2))
        self.upscale_layers = nn.Sequential(*upscale)

        self.output_conv = nn.Conv2d(base_channel, ou_ch, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_conv(x)
        residual = x
        x = self.residual_layers(x)
        x = self.mid_conv(x)
        x = torch.add(x, residual)
        x = self.upscale_layers(x)
        x = self.output_conv(x)
        return x
