import os
import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import losses
import argparse
from dataset import load_dataset
from torch.utils.data import DataLoader
from model import RDDBNetA, RDDBNetB, SRDenseNetA, SRDenseNetB
import itertools
import numpy as np
from utils import Logger


class CasSRC(object):
    """
        cas-SR-Colorizartion
    """

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # define networks (both Generators and discriminators)
        self.netG_A2C = RDDBNetA(1, 1, 64, nb=3, mode=opt.mode).to(opt.device)
        self.netG_C2B = RDDBNetA(1, 3, 64, nb=3, mode='x1').to(opt.device)
        # define loss functions
        self.criterionSR = losses.L1Loss()
        self.criterionC = losses.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG_A2C.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netG_C2B.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def update_lr(self, opt):
        for optimizer in self.optimizers:
            if opt.lr_policy == 'linear':
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_l)
                scheduler.step()
            elif opt.lr_policy == 'step':
                scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                scheduler.step()
            elif opt.lr_policy == 'plateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01,
                                                           patience=5)
                scheduler.step(opt.matrix)
            elif opt.lr_policy == 'cosine':
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs, eta_min=0)
                scheduler.step()
            else:
                return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forwardSR(self, realB):
        self.real_B = realB
        if self.opt.mode == "x2":
            sf = 2
        else:
            sf = 4
        # Y = 0.2125 R + 0.7154 G + 0.0721 B [RGB2Gray, 3=>1 ch]
        self.real_BC = 0.2125 * self.real_B[:,:1,:,:] + \
                       0.7154 * self.real_B[:,1:2,:,:] + \
                       0.0721 * self.real_B[:,2:3,:,:]
        self.real_BA = F.interpolate(self.real_BC, scale_factor=1. / sf)
        self.fake_BC = self.netG_A2C(self.real_BA)
#         print("realB =>", self.real_B.size())
#         print("realBA =>", self.real_BA.size())
#         print("realBC =>", self.real_BC.size())
#         print("fakeBC =>", self.fake_BC.size())

    def forwardC(self):
        self.fake_BB = self.netG_C2B(self.real_BC)
#         print("fakeBB =>", self.fake_BB.size())

    def transfer(self, realA):
        self.real_A = realA
        self.fake_AC = self.netG_A2C(self.real_A)
        self.fake_AB = self.netG_C2B(self.fake_AC)
#         print("fakeAC =>", self.fake_AC.size())
#         print("fakeAC =>", self.fake_AC.size())
#         print("fakeAB =>", self.fake_AB.size())

    def backward_D(self):
        self.loss_D = self.criterionC(self.fake_BB, self.real_B)
        self.loss_D.backward()

    def backward_G(self):
        self.loss_G = self.criterionSR(self.fake_BC, self.real_BC)
        self.loss_G.backward()

    def optimize_parameters(self, realA, realB):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forwardSR(realB)
        # G_A and G_B
#         self.set_requires_grad([self.netG_A2C], True)
#         self.set_requires_grad([self.netG_C2B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # forward
        self.forwardC()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # transfer B => A
        self.transfer(realA)


class params(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = 1e-4
        self.beta1 = 0.5
        self.batch_size = 1
        self.num_works = 2
        self.num_epochs = 25
        self.n_epochs_decay = 100
        self.matrix = 0
        self.lr_policy = 'cosine'


if __name__ == '__main__':
    parse =  argparse.ArgumentParser()
    parse.add_argument('--mode', type=str)
    args = parse.parse_args()
    # Hyperparameters
    opt = params()
    opt.mode = args.mode
    ### Build model
    model = CasSRC(opt)
    ### Data preparation
    trainset, valset, testset = load_dataset('Sat2Aer{}'.format(opt.mode))
    print("Starting Training Loop...")
    # For each epoch
    logger = Logger(len(trainset), opt.num_epochs)
    for epoch in range(1, opt.num_epochs+1):
        # setup data loader
        data_loader = DataLoader(trainset, opt.batch_size, num_workers=opt.num_works,
                                 shuffle=True, pin_memory=True, )
        model.update_lr(opt)
        for idx, sample in enumerate(data_loader):
            realA = sample['src'].to(opt.device)
            realB = sample['tar'].to(opt.device)
            model.optimize_parameters(realA, realB)
            ### 可视化 ###
            if idx % 20 == 0:
                logger.log(
                    nepoch=epoch,
                    niter=idx,
                    losses={'loss_G': model.loss_G.item(),
                            'loss_D': model.loss_D.item()},
                    images={
                        'real_A' : model.real_A, 
                        'fake_AC': model.fake_AC,
                        'fake_AB': model.fake_AB,
                        'real_BA': model.real_BA,
                        'real_BC': model.real_BC,
                        'real_B' : model.real_B, 
                        'fake_BC': model.fake_BC, 
                        'fake_BB': model.fake_BB,
                           }
                )

             ### 可视化 ###
        if epoch % 5 == 0:
            netGA = './checkpoints/netG_A2C_%s_%04d.pth' % (opt.mode, epoch)
            netGB = './checkpoints/netG_C2B_%s_%04d.pth' % (opt.mode, epoch)
            torch.save(model.netG_A2C.state_dict(), netGA)
            torch.save(model.netG_C2B.state_dict(), netGB)
            import os
            os.system('python testCas.py --netGA {} --netGB {}'.format(netGA, netGB))
