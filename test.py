import os
import sys

import torch
import random
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
import losses
from dataset import load_dataset
from torch.utils.data import DataLoader
from SRC64 import RDDBNetA, RDDBNetB, NLayerDiscriminator
import itertools
import numpy as np
from utils import Logger



class params(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = 1e-5
        self.beta1 = 0.5
        self.batch_size = 1
        self.num_works = 2
        self.num_epochs = 25
        self.pool_size = 4
        self.lambda_identity = 0
        self.lambda_A = 10
        self.lambda_B = 10
        self.n_epochs_decay = 100
        self.matrix = 0
        self.lr_policy = 'cosine'


if __name__ == '__main__':
    # Hyperparameters
    opt = params()
    ### Build model

    netG_A2B = RDDBNetB(1, 3, 64, nb=1).to(opt.device)
    netG_B2A = RDDBNetA(3, 1, 64, nb=1).to(opt.device)

    # load check point
    checkpoint = torch.load('./checkpoints/netG_A2B_0000.pth')
    netG_A2B.load_state_dict(checkpoint)
    checkpoint = torch.load('./checkpoints/netG_B2A_0010.pth')
    netG_B2A.load_state_dict(checkpoint)
    netG_A2B.eval()
    netG_B2A.eval()

    if not os.path.exists('result/B_A_64'):
        os.makedirs('result/B_A_64')
    if not os.path.exists('result/A_B_64'):
        os.makedirs('result/A_B_64')

    ### Data preparation
    trainset, valset, testset = load_dataset('Sat2Aerx4')
    print("Starting test Loop...")
    # For each epoch
    logger = Logger(len(testset), opt.num_epochs)
    # setup data loader
    data_loader = DataLoader(testset, opt.batch_size, num_workers=opt.num_works,
                             shuffle=True, pin_memory=True, )
    for idx, sample in enumerate(data_loader):
        realA = sample['src'].to(opt.device)
        realB = sample['tar'].to(opt.device)
        fake_B = netG_A2B(realA)
        fake_A = netG_B2A(realB)
        torchvision.utils.save_image(fake_A, 'result/B_A_64/test_00%04d.png' % (idx))
        torchvision.utils.save_image(fake_B, 'result/A_B_64/test_00%04d.png' % (idx))
        sys.stdout.write('\rGenerated images %04d of %04d' % (idx, len(data_loader)))
