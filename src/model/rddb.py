import functools
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_deconv_params(upscale_factor):
    """
    args:
        ratio:(str) upsample level
        H_out =(H_in−1)×stride[0]−2×pad[0]+ksize[0]+output_padding[0]
    """
    if upscale_factor == 2:
        kernel_size = 2
        stride = 2
    elif upscale_factor == 4:
        kernel_size = 2
        stride = 4
    elif upscale_factor == 8:
        kernel_size = 4
        stride = 8
    output_padding = (stride - kernel_size)
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


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5, self).__init__()
        # gc = growth channeal, image = 256
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)  # 64 -32,256
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)  # 96, 32, 256
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1,
                               bias=bias)  # 128, 32, 256
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1,
                               bias=bias)  # 160, 32, 256
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1,
                               bias=bias)  # 192, 32, 256
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, lemda=0.2):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * lemda + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5(nf, gc)
        self.RDB2 = ResidualDenseBlock_5(nf, gc)
        self.RDB3 = ResidualDenseBlock_5(nf, gc)

    def forward(self, x, lemda=0.2):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * lemda + x


class RDDBNet(nn.Module):
    def __init__(self, in_ch, ou_ch, upscale_factor, nf=64, nb=3, gc=32):
        super(RDDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upscale_factor = upscale_factor
        upscale = []
        for _ in range(int(math.log2(upscale_factor))):
            upscale.append(deconv(nf, nf, upscale_factor=2))
            upscale.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.upscale_layers = nn.Sequential(*upscale)
        self.conv_last = nn.Conv2d(nf, ou_ch, 3, 1, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        if self.upscale_factor != 1:
            fea = self.upscale_layers(fea)
        out = self.conv_last(fea)
        return out
