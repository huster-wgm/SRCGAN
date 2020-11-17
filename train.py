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

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(device))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's checkpoints and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction checkpoints from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class SRCycleGAN(object):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
 """

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = RDDBNetB(1,3,64, nb=1).to(opt.device)
        self.netG_B = RDDBNetA(3,1,64, nb=1).to(opt.device)

        self.netD_A = NLayerDiscriminator(3,64,3).to(opt.device)
        self.netD_B = NLayerDiscriminator(1,64,3).to(opt.device)
        self.fake_A_pool = ImagePool(opt.pool_size)
        self.fake_B_pool = ImagePool(opt.pool_size)
        # define loss functions
        self.criterionGAN = GANLoss(gan_mode='lsgan', device=opt.device)  # define GAN loss.
        self.criterionCycle = losses.DSSIMLoss()
        self.criterionIdt = losses.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def update_lr(self,opt):
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

    def forward(self, realA, realB):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_A = realA
        self.real_B = realB
        self.fake_B = self.netG_A(self.real_A)   # G_A(A) 256
        self.recl_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)   # G_B(B) 64
        self.recl_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B) # real B 不是64 么， 怎么放进GA 256 - 64 生成器中？
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.recl_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.recl_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self, realA, realB):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward(realA, realB)      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

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
        self.n_epochs_decay=100
        self.matrix=0
        self.lr_policy='cosine'
        
if __name__ == '__main__':
    # Hyperparameters
    opt = params()
    ### Build model
    model = SRCycleGAN(opt)
    ### Data preparation
    trainset, valset, testset = load_dataset('Sat2Aerx4')
    print("Starting Training Loop...")
    # For each epoch
    logger = Logger(len(trainset), opt.num_epochs)
    for epoch in range(0, opt.num_epochs):
        # setup data loader
        data_loader = DataLoader(trainset, opt.batch_size, num_workers=opt.num_works,
                                 shuffle=True, pin_memory=True,)
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
                            'loss_G_identity': 0.0,
                            'loss_G_GAN': (model.loss_G_A.item() + model.loss_G_B.item()),
                            'loss_G_cycle': (model.loss_cycle_A.item() + model.loss_cycle_B.item()),
                            'loss_D': (model.loss_D_A.item() + model.loss_D_B.item())},
                    images={'real_A': model.real_A, 'real_B': model.real_B,
                            'fake_A': model.fake_A, 'fake_B': model.fake_B,
                            'recl_A': model.recl_A, 'recl_B': model.recl_B,
                           })
        ### 可视化 ###
        if epoch % 5 == 0:
            torch.save(model.netG_A.state_dict(), './checkpoints/netG_A2B_%04d.pth' % epoch)
            torch.save(model.netG_B.state_dict(), './checkpoints/netG_B2A_%04d.pth' % epoch)

