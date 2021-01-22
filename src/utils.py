import random
import time
import datetime
import sys
import os
import torch
import cv2
from visdom import Visdom
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import numpy as np

dsize = (256, 256)


def tensor2img(tensor, mode="RGB"):
    if mode == "RGB":
        img = tensor[0].cpu().numpy().transpose((1,2,0))
    #     img += 127
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        img = (img * 255).astype(np.uint8)
    else: # lab2rgb
        lab = tensor[0].cpu().numpy().transpose((1,2,0))
        lab[:,:,:1] = lab[:,:,:1] * 100
        lab[:,:,1:] = lab[:,:,1:] * 255 - 128
        img = (lab2rgb(lab.astype("float64")) * 255).astype("uint8")
    img = cv2.resize(img, dsize=dsize).transpose((2,0,1))
#     print(img.shape)
    return img


class Logger():
    def __init__(self, n_iters, n_epochs):
        self.viz = Visdom()
        self.n_iters = n_iters
        self.n_epochs = n_epochs
        self.init_time = time.time()

    def log(self, nepoch, niter, losses=None, images=None, ver='G2RGB'):
        period = time.time() - self.init_time
        sys.stdout.write('\n Epoch %02d [%04d/%04d] >> ' % 
                         (nepoch, niter, self.n_iters))
        for k, v in losses.items():
            sys.stdout.write('%s: %.3f | ' % (k, v))

        iters_done = self.n_iters * (nepoch -1) + niter
        iters_left = self.n_iters * self.n_epochs - iters_done
        eta = iters_left / iters_done * period
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds = eta)))

        # Draw imgs
        for k, v in images.items():
            mode = "RGB"
            if k in ['fake_AB', 'real_B', 'fake_BB'] and ver=="G2LAB":
                mode = 'LAB'
            img = tensor2img(v.data, mode)
            self.viz.image(img, 
                           win = k,
                           opts = {'title' : k})


