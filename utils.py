import random
import time
import datetime
import sys

import torch
import cv2
from visdom import Visdom
import numpy as np

dsize = (256, 256)

def tensor2image(tensor):
    image = tensor[0].cpu().float().numpy()*255
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    image = image.astype(np.uint8).transpose((1,2,0))
    image = cv2.resize(image, dsize=dsize).transpose((2,0,1))
#     print(image.shape)
    return image

class Logger():
    def __init__(self, n_iters, n_epochs):
        self.viz = Visdom()
        self.n_iters = n_iters
        self.n_epochs = n_epochs
        self.init_time = time.time()

    def log(self, nepoch, niter, losses=None, images=None):
        period = time.time() - self.init_time
        sys.stdout.write('\n Epoch => %03d/%03d [%04d/%04d] >> | ' % 
                         (nepoch, self.n_epochs, niter, self.n_iters))
        for k, v in losses.items():
            sys.stdout.write('%s: %.4f | ' % (k, v))

        iters_done = self.n_iters * nepoch + niter + 1
        iters_left = self.n_iters * self.n_epochs - iters_done
        sys.stdout.write('ETA: %s' % 
                         (datetime.timedelta(seconds=iters_left/iters_done*period)))

        # Draw images
        for k, v in images.items():
            self.viz.image(tensor2image(v.data), 
                           win = k,
                           opts = {'title' : k}
                          )


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
        

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


