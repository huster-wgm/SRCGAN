import os
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
from utils import Logger
from skimage.color import lab2rgb, rgb2lab, rgb2gray


DIR = os.path.dirname(os.path.abspath(__file__))
Check_DIR = os.path.join(DIR, '../checkpoints/')


class params(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.num_works = 2


def tensor2img(tensor, ver='RGB'):
    if ver == "RGB":
        img = tensor.detach()[0].cpu().numpy().transpose((1,2,0))
        img = (img * 255).astype("uint8")
        if img.shape[2] == 1:
            img = img[:,:,0]
    else:
        lab = tensor.detach()[0].cpu().numpy().transpose((1,2,0))
        lab[:,:,:1] = lab[:,:,:1] * 100
        lab[:,:,1:] = lab[:,:,1:] * 255 - 128
        img = (lab2rgb(lab.astype("float64")) * 255).astype("uint8")
    return img


if __name__ == '__main__':
    parse =  argparse.ArgumentParser()
    parse.add_argument('--netGA', type=str)
    parse.add_argument('--netGB', type=str)
    args = parse.parse_args()
    # Hyperparameters
    opt = params()
    checkA = os.path.basename(args.netGA).split('.pth')[0].split('_')
    checkB = os.path.basename(args.netGB).split('.pth')[0].split('_')
    ### Data preparation
    trainset, valset, testset = load_dataset('Sat2Aerx1', ver="G2LAB")
    ### makedirs
    save_dirA = './result/A_'+"_".join([checkA[0], checkA[2], checkA[3]])
    save_dirB = './result/B_'+"_".join([checkA[0], checkA[2], checkA[3]])
    if not os.path.exists(save_dirA) or not os.path.exists(save_dirB):
        os.makedirs(save_dirA)
        os.makedirs(save_dirB)
    ### Build model
    netG_A2C = eval(checkA[0].replace('@G2LAB', ''))(1, 1, int(checkA[2][1])).to(opt.device)
    netG_C2B = eval(checkB[0].replace('@G2LAB', ''))(1, 2).to(opt.device)
    # load check point
    netG_A2C.load_state_dict(torch.load(os.path.join(Check_DIR, os.path.basename(args.netGA))))
    netG_C2B.load_state_dict(torch.load(os.path.join(Check_DIR, os.path.basename(args.netGB))))
    netG_A2C.eval()
    netG_C2B.eval()
    print("Starting test Loop...")
    # setup data loader
    data_loader = DataLoader(testset, opt.batch_size, num_workers=opt.num_works,
                             shuffle=False, pin_memory=True, )
    evaluators = [metrics.MSE(), metrics.PSNR(), metrics.AE(), metrics.SSIM()]
    performs = [[] for i in range(len(evaluators))]
    for idx, sample in enumerate(data_loader):
        realA = sample['src'].to(opt.device)
        realB = sample['tar'].to(opt.device)
        realBC = realB[:,:1,:,:]
        sf = int(checkA[2][1])
        realBA = nn.functional.interpolate(realBC, scale_factor=1. / sf)
#         realBA = nn.functional.interpolate(realBA, scale_factor=sf)
        realAA = nn.functional.interpolate(realA, scale_factor=1. / sf)
        fake_AC = netG_A2C(realAA)
        fake_AB = netG_C2B(fake_AC)
        fake_BC = netG_A2C(realBA)
        fake_BB = netG_C2B(fake_BC)
        fakeAB = torch.cat([fake_AC, fake_AB], dim=1)
        fakeBB = torch.cat([fake_BC, fake_BB], dim=1)
        # calc performances
        acc = ""
        for i, evaluator in enumerate(evaluators):
            p = evaluator(fakeBB.detach(), realB.detach()).item()
            acc += " {}:{:0.2f};".format(repr(evaluator), p)
            performs[i].append(p)
        # save generate image
        imsave(save_dirA+'/%s' % (testset.datalist[idx]), tensor2img(fakeAB, 'LAB'))
        imsave(save_dirB+'/%s' % (testset.datalist[idx]), tensor2img(fakeBB, 'LAB'))
        sys.stdout.write('\rGenerated %s (%04d / %04d) >> %s' % 
                         (testset.datalist[idx], idx, len(data_loader), acc))
    # save performances
    performs = [(sum(p) / len(p)) for p in performs]
    performs = pd.DataFrame([[time.strftime("%h_%d"), 
                              os.path.basename(args.netGA).split('.pth')[0]] + performs],
                              columns=['time', 'checkpoint'] + [repr(x) for x in evaluators])
    # save performance
    log_path = os.path.join('./result', "Performs.csv")
    if os.path.exists(log_path):
        perform = pd.read_csv(log_path)
    else:
        perform = pd.DataFrame([])
    perform = perform.append(performs, ignore_index=True)
    perform.to_csv(log_path, index=False, float_format="%.3f")
