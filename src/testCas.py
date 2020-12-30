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
    return image


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
    trainset, valset, testset = load_dataset('Sat2Aerx1')
    ### makedirs
    save_dirA = './result/A_'+"_".join([checkA[0], checkA[2], checkA[3]])
    save_dirB = './result/B_'+"_".join([checkA[0], checkA[2], checkA[3]])
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
    print("Starting test Loop...")
    # setup data loader
    data_loader = DataLoader(testset, opt.batch_size, num_workers=opt.num_works,
                             shuffle=True, pin_memory=True, )
    evaluators = [metrics.MSE(), metrics.PSNR(), metrics.AE(), metrics.SSIM()]
    performs = [[] for i in range(len(evaluators))]
    y_pred, y = [], []
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
        fake_AB = netG_C2B(netG_A2C(realAA))
        fake_BB = netG_C2B(netG_A2C(realBA))
        y.append(realB.detach().cpu())
        y_pred.append(fake_BB.detach().cpu())
        imsave(save_dirA+'/test_%06d.png' % (idx), tensor2image(fake_AB))
        imsave(save_dirB+'/test_%06d.png' % (idx), tensor2image(fake_BB))
        sys.stdout.write('\rGenerated images %06d of %06d' % (idx, len(data_loader)))
    # calc performances
    y = torch.cat(y, 0)
    y_pred = torch.cat(y_pred, 0)
    for idx, evaluator in enumerate(evaluators):
        performs[idx].append(evaluator(y_pred, y).item())
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
