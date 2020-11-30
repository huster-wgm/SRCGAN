#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import numpy as np
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import math

eps = 1e-6


class SSIM(object):
    '''
    modified from https://github.com/jorge-pessoa/pytorch-msssim
    '''
    def __init__(self, des="structural similarity index"):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=6, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)
        # print(window.size())
        # print(y_pred.size())
        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return "L1"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

class L1Loss3D(nn.Module):
    def __init__(self):
        super(L1Loss3D, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return "L13D"

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append(self.criterion(output[:,:,f,:,:], target[:,:,f,:,:]))
        return sum(loss) / len(loss)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
    
    def __repr__(self):
        return "MSE"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def __repr__(self):
        return "PSNR"

    def forward(self, output, target):
        mse = self.criterion(output, target)
        loss = 10 * torch.log10(1.0 / mse)
        return loss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterionBinary = nn.BCELoss(size_average=True)
        self.criterionMulti = nn.NLLLoss(size_average=True)

    def __repr__(self):
        return "CE"

    def forward(self, output, target):
        if target.shape[1] == 1:
            # binary cross enthropy
            loss = self.criterionBinary(output, target)
        else:
            # multi-class cross enthropy
            target = torch.argmax(target, dim=1).long()
            loss = self.criterionMulti(torch.log(output), target)
        return loss


class DSSIMLoss(nn.Module):
    def __init__(self):
        super(DSSIMLoss, self).__init__()
        self.criterion = SSIM()

    def __repr__(self):
        return "DSSIM"

    def forward(self, output, target):
        loss = (1. - self.criterion(output, target)) / 2.
        return loss


class DSSIMLoss3D(nn.Module):
    def __init__(self):
        super(DSSIMLoss3D, self).__init__()
        self.criterion = SSIM()

    def __repr__(self):
        return "DSSIM3D"

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append((1. - self.criterion(output[:,:,f,:,:], target[:,:,f,:,:])) / 2.)
        return sum(loss) / len(loss)


class NearestSelector(object):
    def __init__(self, shift=2, stride=1, criter='l1'):
        self.shift = shift
        self.stride = stride
        self.criter = criter

    def __repr__(self):
        return "NS"

    @staticmethod
    def unravel_index(tensor, cols):
        """
        args:
            tensor : 2D tensor, [nb, rows*cols]
            cols : int
        return 2D tensor nb * [rowIndex, colIndex]
        """
        index = torch.argmin(tensor, dim=1).view(-1,1)
        rIndex = index / cols
        cIndex = index % cols
        minRC = torch.cat([rIndex, cIndex], dim=1)
        # print("minRC", minRC.shape, minRC)
        return minRC

    def shift_diff(self, output, target, crop_row, crop_col):
        diff = []
        for i in range(0, 2 * self.shift):
            for j in range(0, 2 * self.shift):
                output_crop = output[:, :, 
                                     self.shift * self.stride: self.shift * self.stride + crop_row,
                                     self.shift * self.stride: self.shift * self.stride + crop_col,]
                target_crop = target[:, :, 
                                     i * self.stride: i * self.stride + crop_row,
                                     j * self.stride: j * self.stride + crop_col,]
                diff_ij = torch.sum(abs(target_crop-output_crop), dim=[1,2,3]).view(-1,1)
                diff.append(diff_ij)
        return torch.cat(diff, dim=1)
        
    def crop(self, output, target):
        nb, ch, row, col = output.shape
        crop_row = row - 2 * self.shift * self.stride
        crop_col = col - 2 * self.shift * self.stride
        diff = self.shift_diff(output.detach(), target.detach(), crop_row, crop_col)
        minRC = self.unravel_index(diff, 2 * self.shift)
        crop = [self.shift * self.stride, self.shift * self.stride + crop_row,
                self.shift * self.stride, self.shift * self.stride + crop_col]
        output_ = output[:,
                         :,
                        crop[0] : crop[1],
                        crop[2] : crop[3]]
        target_ = torch.zeros(*output_.shape).to(target.device)
        for idx, (minR, minC) in enumerate(minRC):
            target_[idx] = target[idx,
                                  :,
                                  minR * self.stride: minR * self.stride + crop_row,
                                  minC * self.stride: minC * self.stride + crop_row]
        return output_, target_

    
class ConLoss(nn.Module):
    """
    Consistency of samples within batch
    """

    def __init__(self):
        super(ConLoss, self).__init__()
        self.criterMSE = nn.MSELoss(size_average=True)

    def __repr__(self):
        return 'ConLoss'

    def forward(self, feats):
        feat_max, _ = torch.max(feats, dim=0)
        feat_min, _ = torch.min(feats, dim=0)
        zeros = torch.zeros(feat_max.shape).to(feats.device)
        return self.criterMSE(torch.abs(feat_max - feat_min), zeros)


class CrossLoss(nn.Module):
    """
    Cross comparison between samples within batch
    """

    def __init__(self):
        super(CrossLoss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return 'CrossLoss'

    def forward(self, output, target):
        nb, ch, row, col = output.shape
        output = output[:nb-1, :, :, :]
        target = target[1:nb, :, :, :]
        return self.criterion(output, target)


class FLoss(nn.Module):
    """
    Focal Loss
    Lin, Tsung-Yi, et al. \
    "Focal loss for dense object detection." \
    Proceedings of the IEEE international conference on computer vision. 2017.
    (modified from https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py)
    """
    def __init__(self, gamma=2., weight=None, size_average=True):
        super(FLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def __repr__(self):
        return 'Focal'

    def _get_weights(self, y_true, nb_ch):
        """
        args:
            y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
            nb_ch : int 
        return [float] weights
        """
        batch_size, img_rows, img_cols = y_true.shape
        pixels = batch_size * img_rows * img_cols
        weights = [torch.sum(y_true==ch).item() / pixels for ch in range(nb_ch)]
        return weights

    def forward(self, output, target):
        output = torch.clamp(output, min=eps, max=(1. - eps))
        if target.shape[1] == 1:
            # binary focal loss
            # weights = self._get_weigthts(target[:,0,:,:], 2)
            alpha = 0.1
            loss = - (1.-alpha) * ((1.-output)**self.gamma)*(target*torch.log(output)) \
              - alpha * (output**self.gamma)*((1.-target)*torch.log(1.-output))
        else:
            # multi-class focal loss
            # weights = self._get_weigthts(torch.argmax(target, dim=1), target.shape[1])
            loss = - ((1.-output)**self.gamma)*(target*torch.log(output))

        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()
        
        
class VGG16Loss(nn.Module):
    def __init__(self, requires_grad=False, cuda=True):
        super(VGG16Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if cuda:
            self.slice1.cuda()
            self.slice2.cuda()
            self.slice3.cuda()
            self.slice4.cuda()

    def __repr__(self):
        return "VGG16"

    def forward(self, output, target):
        nb, ch, row, col = output.shape
        if ch == 1:
            output = torch.cat([output, output, output], dim=1)
            target = torch.cat([target, target, target], dim=1)
        ho = self.slice1(output)
        ht = self.slice1(target)
        h_relu1_2_loss = self.criterion(ho,ht)
        ho = self.slice2(ho)
        ht = self.slice2(ht)
        h_relu2_2_loss = self.criterion(ho,ht)
        ho = self.slice3(ho)
        ht = self.slice3(ht)
        h_relu3_3_loss = self.criterion(ho,ht)
        ho = self.slice4(ho)
        ht = self.slice4(ht)
        h_relu4_3_loss = self.criterion(ho,ht)
        return sum([h_relu1_2_loss, h_relu2_2_loss, h_relu3_3_loss, h_relu4_3_loss]) / 4


class VGG16Loss3D(nn.Module):
    def __init__(self, requires_grad=False, cuda=True):
        super(VGG16Loss3D, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if cuda:
            self.slice1.cuda()
            self.slice2.cuda()
            self.slice3.cuda()
            self.slice4.cuda()

    def __repr__(self):
        return "VGG163D"

    def forward2d(self, output, target):
        nb, ch, row, col = output.shape
        if ch == 1:
            output = torch.cat([output, output, output], dim=1)
            target = torch.cat([target, target, target], dim=1)
        ho = self.slice1(output)
        ht = self.slice1(target)
        h_relu1_2_loss = self.criterion(ho,ht)
        ho = self.slice2(ho)
        ht = self.slice2(ht)
        h_relu2_2_loss = self.criterion(ho,ht)
        ho = self.slice3(ho)
        ht = self.slice3(ht)
        h_relu3_3_loss = self.criterion(ho,ht)
        ho = self.slice4(ho)
        ht = self.slice4(ht)
        h_relu4_3_loss = self.criterion(ho,ht)
        return sum([h_relu1_2_loss, h_relu2_2_loss, h_relu3_3_loss, h_relu4_3_loss]) / 4

    def forward(self, output, target):
        nb, ch, frame, row, col = output.shape
        loss = []
        for f in range(frame):
            loss.append(
                self.forward2d(output[:,:,f,:,:], target[:,:,f,:,:]))
        return sum(loss) / len(loss)


if __name__ == "__main__":
    for ch in [3, 1]:
        for cuda in [True, False]:
            batch_size, img_row, img_col = 32, 24, 24
            y_true = torch.rand(batch_size, ch, img_row, img_col)
            y_pred = torch.rand(batch_size, ch, img_row, img_col)
            if cuda:
                y_pred = y_pred.cuda()
                y_true = y_true.cuda()

            print('#'*20, 'Test on cuda : {} ; size : {}'.format(cuda, y_true.size()))

            y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
            criterion = L1Loss()
            print('\t gradient bef : {}'.format(y_pred_.grad))
            loss = criterion(y_pred_, y_true_)
            loss.backward()
            print('\t gradient aft : {}'.format(y_pred_.grad.shape))
            print('{} : {}'.format(repr(criterion), loss.item()))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = CELoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = MSELoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = FLoss()
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

#             y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
#             criterion = VGG16Loss(cuda=cuda)
#             loss = criterion(y_pred_, y_true_)
#             loss.backward()
#             print('{} : {}'.format(repr(criterion), loss.item()))
#             print('\t gradient : {}'.format(y_pred_.grad.shape))

            y_pred_, y_true_ = y_pred.clone().requires_grad_(), y_true.clone()
            selector = NearestSelector()
            y_pred_near, y_true_near = selector.crop(y_pred_, y_true_)
            criterion = L1Loss()
            print('\t gradient bef : {}'.format(y_pred_.grad))
            loss = criterion(y_pred_near, y_true_near)
            loss.backward()
            print('\t gradient aft : {}'.format(y_pred_.grad.shape))
            print('{}-near : {}'.format(repr(criterion), loss.item()))

