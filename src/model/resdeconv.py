#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import numpy as np
import torch
import torch.nn as nn

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                     stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_deconv_params(ratio="x2"):
    """
    args:
        ratio:(str) upsample level
        H_out =(H_in−1)×stride[0]−2×pad[0]+ksize[0]+output_padding[0] 
    """
    if ratio == "x2":
        kernel_size = 2
        stride = 2
    elif ratio == "x4":
        kernel_size = 2
        stride = 4
    elif ratio == "x8":
        kernel_size = 4
        stride = 8
    output_padding = (stride-kernel_size)
    return kernel_size, stride, output_padding


def deconv(in_planes, out_planes, ratio="x2"):
    """2d deconv"""
    kernel_size, stride, opad = get_deconv_params(ratio)
    # print("DECONV", kernel_size, stride, opad)
    return nn.ConvTranspose2d(in_planes, out_planes, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=0, 
                              bias=False,
                              output_padding=opad,
                              groups=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, BNmode='IN'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if BNmode == 'BN':
            self.bn1 = nn.BatchNorm2d(planes)
        elif BNmode == 'IN':
            self.bn1 = nn.InstanceNorm2d(planes)
        elif BNmode == 'GN':
            self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if BNmode == 'BN':
            self.bn2 = nn.BatchNorm2d(planes)
        elif BNmode == 'IN':
            self.bn2 = nn.InstanceNorm2d(planes)
        elif BNmode == 'GN':
            self.bn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResDeconv(nn.Module):
    """Constructs a 
        ResNet-18 layers = [2, 2, 2, 2]
        ResNet-34 layers = [3, 4, 6, 3]

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    def __init__(self, src_ch=1, tar_ch=3, block=BasicBlock, layers=[2, 2, 2, 2], instance=True):
        super(ResDeconv, self).__init__()
        self.src_ch = src_ch
        if isinstance(tar_ch, list):
            tar_ch = sum(tar_ch)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if not instance else nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, instance=instance)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, instance=instance)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, instance=instance)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, instance=instance)

        # Left arm
        self.deconv10 = deconv(512, 256, ratio="x2")
        self.inplanes = 256
        self.upRes1 = self._make_layer(block, 256, layers[2], instance=instance)
        self.deconv11 = deconv(256, 128, ratio="x2")
        self.inplanes = 128
        self.upRes2 = self._make_layer(block, 128, layers[1], instance=instance)
        self.deconv12 = deconv(128,  64, ratio="x2")
        self.inplanes = 64
        self.upRes3 = self._make_layer(block, 64, layers[0], instance=instance)
        self.deconv13 = deconv( 64,  64, ratio="x2")
        self.pred = nn.Conv2d(64, tar_ch, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, instance=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion) if not instance else nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # forward network
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2 = self.relu(x1)
        # x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        # generate left
        x7 = self.deconv10(x6)
        # x8L = x7L + x5 # add x7 and x5
        x9 = self.upRes1(x7)
        x10 = self.deconv11(x9)
        # x11L = x10L + x4 # add x11 and x4
        x12 = self.upRes2(x10)
        x13 = self.deconv12(x12)
        # x14L = x13L + x3 # add x13 and x3
        x15 = self.upRes3(x13)
        x16 = self.deconv13(x15)
        return self.pred(x16)
