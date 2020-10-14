import random
import time
import datetime
import sys

import torch
from visdom import Visdom
import numpy as np

def tensor2image(tensor):
    image = tensor[0].cpu().float().numpy()*255
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, interval):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.stepes_epoch = batches_epoch
        self.interval = interval
        self.epoch = 1
        self.step = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('Epoch %03d/%03d [%04d/%04d] -- \n ' % (self.epoch, self.n_epochs, self.step*self.interval, self.stepes_epoch))

        for i, loss_name in enumerate(losses.keys()):
            sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]))

        batches_done = self.stepes_epoch*(self.epoch - 1) + self.interval * self.step
        batches_left = self.stepes_epoch*(self.n_epochs - self.epoch + 1) - self.interval * self.step
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.step % self.stepes_epoch) == 0:
            # Plot losses
            # for loss_name, loss in self.losses.items():
            #     # if loss_name not in self.loss_windows:
            #     #     self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.step]),
            #     #                                                     opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
            #     # else:
            #     #     self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.step]), win=self.loss_windows[loss_name], update='append')
            #     # # Reset losses for next epoch
            #     # self.losses[loss_name] = 0.0

            self.epoch += 1
            self.step = 1
            # sys.stdout.write('\n')
        else:
            self.step += 1

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


