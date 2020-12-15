import os
import sys
import cv2
import torch
import random
import torchvision
import torch.nn as nn
import argparse
from dataset import load_dataset
from torch.utils.data import DataLoader
from model import RDDBNetA, RDDBNetB, NLayerDiscriminator
import itertools
from skimage.io import imsave
from utils import Logger


class params(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.num_works = 2


def tensor2image(tensor):
    import numpy as np
    image = tensor.detach()[0].cpu().float().numpy()*255
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
    checkA = os.path.basename(args.netGA).split('.pth')[0]
    checkB = os.path.basename(args.netGB).split('.pth')[0]
    ### Data preparation
    if 'x2' in checkA:
        opt.mode = 'x2'
        trainset, valset, testset = load_dataset('Sat2Aerx2')
    elif 'x4' in checkA:
        opt.mode = 'x4'
        trainset, valset, testset = load_dataset('Sat2Aerx4')
    ### makedirs
    if not os.path.exists('result/A'+opt.mode):
        os.makedirs('result/A'+opt.mode)
    if not os.path.exists('result/B'+opt.mode):
        os.makedirs('result/B'+opt.mode)
    ### Build model
    netG_A2C = RDDBNetA(1, 1, 64, nb=3, mode= opt.mode).to(opt.device)
    netG_C2B = RDDBNetA(1, 3, 64, nb=3, mode= 'x1').to(opt.device)
    # load check point
    netG_A2C.load_state_dict(torch.load(args.netGA))
    netG_C2B.load_state_dict(torch.load(args.netGB))
    netG_A2C.eval()
    netG_C2B.eval()
    print("Starting test Loop...")
    # setup data loader
    data_loader = DataLoader(testset, opt.batch_size, num_workers=opt.num_works,
                             shuffle=True, pin_memory=True, )
    for idx, sample in enumerate(data_loader):
        realA = sample['src'].to(opt.device)
        realB = sample['tar'].to(opt.device)
        # Y = 0.2125 R + 0.7154 G + 0.0721 B [RGB2Gray, 3=>1 ch]
        realBC = 0.2125 * realB[:,:1,:,:] + \
                 0.7154 * realB[:,1:2,:,:] + \
                 0.0721 * realB[:,2:3,:,:]
        if opt.mode == "x2":
            sf = 2
        elif opt.mode == "x4":
            sf = 4

        realBA = nn.functional.interpolate(realBC, scale_factor=1. / sf)
        fake_AB = netG_C2B(netG_A2C(realA))
        fake_BB = netG_C2B(netG_A2C(realBA))
        imsave('result/A'+opt.mode+'/test_%06d.png' % (idx), tensor2image(fake_AB))
        imsave('result/B'+opt.mode+'/test_%06d.png' % (idx), tensor2image(fake_BB))
        sys.stdout.write('\rGenerated images %06d of %06d' % (idx, len(data_loader)))

