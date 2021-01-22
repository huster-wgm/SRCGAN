import os
import cv2
import sys
import torch
import random
import torchvision
import torch.nn as nn
import argparse
import metrics
import time
import numpy as np
import pandas as pd
from dataset import load_dataset
from torch.utils.data import DataLoader
from model import *
from skimage.io import imsave

dsize = (256, 256)


class params(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.num_works = 2


def tensor2image(tensor):
    image = tensor.detach()[0].cpu().float().numpy()*255
#     image += 127
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    image = image.astype(np.uint8).transpose((1,2,0))
#     print(image.shape)
    if image.shape[0] != dsize[0]:
        image = cv2.resize(image, dsize)
    return image


def add_barrier(img, spaces=[5, 10]):
    """
    args:
        img: (ndarray) in [img_rows, img_cols, channels], dtype as unit8
        spaces: (int) pixels of spaces
    return:
        img: (ndarray) processed img
    """
    img = add_color_bar(img, spaces[0], 0)
    img = add_color_bar(img, spaces[1], 255)
    return img


def add_color_bar(img, space, cv):
    """
    args:
        img: (ndarray) in [img_rows, img_cols, channels], dtype as unit8
        space: (int) pixels of space
        cv: (int) color value in [0, 255]
    return:
        tmp_img: (ndarray) processed img
    """
    assert len(img.shape) == 3, "img should be 3D"
    img_rows, img_cols, channels = img.shape
    tmp_img = np.ones((img_rows + 2 * space,
                       img_cols + 2 * space,
                       channels), np.uint8) * cv

    tmp_img[space: space + img_rows,
            space: space + img_cols] = img
    return tmp_img


def patch2vis(patches):
    imgs = []
    for patch in patches:
        img = tensor2image(patch)
        img = add_barrier(img)
        imgs.append(img)
    vis = np.concatenate(imgs, axis=1)
    return vis


if __name__ == '__main__':
    parse =  argparse.ArgumentParser()
    parse.add_argument('--netGA', type=str)
    parse.add_argument('--netGB', type=str)
    parse.add_argument('--threshold', type=float)
    args = parse.parse_args()
    # Hyperparameters
    opt = params()
    checkA = os.path.basename(args.netGA).split('.pth')[0].split('_')
    checkB = os.path.basename(args.netGB).split('.pth')[0].split('_')
    ### Data preparation
    trainset, valset, testset = load_dataset('Sat2Aerx1')
    ### makedirs
    save_dirA = './visResult/A_'+"_".join([checkA[0], checkA[2], checkA[3]])
    save_dirB = './visResult/B_'+"_".join([checkA[0], checkA[2], checkA[3]])
    if not os.path.exists(save_dirA) or not os.path.exists(save_dirB):
        os.makedirs(save_dirA)
        os.makedirs(save_dirB)
    ### Build model
    netG_A2C = eval(checkA[0])(1, 1, int(checkA[2][1])).to(opt.device)
    netG_C2B = eval(checkB[0])(1, 3).to(opt.device)
    # load check point
    netG_A2C.load_state_dict(torch.load(args.netGA))
    netG_C2B.load_state_dict(torch.load(args.netGB))
    netG_A2C.eval()
    netG_C2B.eval()
    print("Starting visualization Loop...")
    # setup data loader
    data_loader = DataLoader(testset, opt.batch_size, num_workers=opt.num_works,
                             shuffle=False, pin_memory=True, )
    evaluator = metrics.PSNR()
    for idx, sample in enumerate(data_loader):
        realA = sample['src'].to(opt.device)
#         realA -= 0.5
        realB = sample['tar'].to(opt.device)
#         realB -= 0.5
        # Y = 0.2125 R + 0.7154 G + 0.0721 B [RGB2Gray, 3=>1 ch]
        realBC = 0.2125 * realB[:,:1,:,:] + \
                 0.7154 * realB[:,1:2,:,:] + \
                 0.0721 * realB[:,2:3,:,:]
        sf = int(checkA[2][1])
        realBA = nn.functional.interpolate(realBC, scale_factor=1. / sf)
#         realBA = nn.functional.interpolate(realBA, scale_factor=sf)
        realAA = nn.functional.interpolate(realA, scale_factor=1. / sf)
        fake_AC = netG_A2C(realAA)
        fake_AB = netG_C2B(fake_AC)
        fake_BC = netG_A2C(realBA)
        fake_BB = netG_C2B(fake_BC)
        perform = evaluator(fake_BB.detach(), realB.detach()).item()
        if perform > args.threshold:
            vis_a = patch2vis([realAA, fake_AC, fake_AB, realB])
            vis_b = patch2vis([realBA, fake_BC, fake_BB, realB])
            imsave(save_dirA+'/test_%06d_comp.png' % (idx), vis_a)
            imsave(save_dirB+'/test_%06d_comp.png' % (idx), vis_b)
            sys.stdout.write('\rSave images (%06d / %06d) PSNR : %0.1f' % 
                             (idx, len(testset), perform))
        else:
            sys.stdout.write('\rSkip images (%06d / %06d) PSNR : %0.1f' % 
                             (idx, len(testset), perform))
