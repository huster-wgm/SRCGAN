#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import argparse
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imsave
from skimage.color import lab2rgb, rgb2lab, rgb2gray

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

DIR = os.path.dirname(os.path.abspath(__file__))
Dataset_DIR = os.path.join(DIR, './dataset/')

class Basic(Dataset):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.split = split
        # file = '{}-nir.txt'.format(self.split)
        file = '{}.txt'.format(self.split)
        with open(os.path.join(Dataset_DIR, self.root, file), 'r') as f:
            self.datalist = [line.strip() for line in f.readlines()]

        self.srcpath = os.path.join(Dataset_DIR, root, "src", '%s')
        self.tarpath = os.path.join(Dataset_DIR, root, "tar", '%s')
        
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    @staticmethod
    def normalize(arr):
        mx = np.max(arr)
        mi = np.min(arr)
        arr = (arr - mi) / (mx - mi)
        return arr

    def _whitespace(self, img, width=5):
        """
        Args:
            img : ndarray [h,w,c]
        """
        row, col, ch = img.shape
        tmp = np.ones((row + 2*width, col + 2*width, ch), "uint8") * 255
        tmp[width:row+width,width:width+col,:] = img
        return tmp
    
    def _g2img(self, arr, whitespace=True):
        """
        Args:
            arr (str): ndarray [h,w,c]
        """
        if arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        img = (arr * 255).astype("uint8")
        if whitespace:
            img = self._whitespace(img)
        return img


    def _rgb2img(self, arr, whitespace=True):
        """
        Args:
            arr (str): ndarray [h,w,c]
        """
        if arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        img = (arr * 255).astype("uint8")
        if whitespace:
            img = self._whitespace(img)
        return img

    def _lab2img(self, lab, whitespace=True):
        """
        Args:
            lab: LAB in [0, 1.0]
        """
        lab[:,:,:1] = lab[:,:,:1] * 100
        lab[:,:,1:] = lab[:,:,1:] * 255 - 128
        img = (lab2rgb(lab.astype("float64")) * 255).astype("uint8")
        if whitespace:
            img = self._whitespace(img)
        return img
    
    def _ab2img(self, l, ab, whitespace=True):
        """
        Args:
            lab: LAB in [0, 1.0]
        """
        lab = np.concatenate([l, ab], axis=-1)
        return self._lab2img(lab, whitespace)

    def _arr2gray(self, arr):
        """
        Args:
            arr: ndarray
        return tensor(L)
        """
        arr = rgb2gray(arr)
        arr = np.expand_dims(arr, axis=-1).transpose((2, 0, 1))
        tensor = torch.from_numpy(arr).float()
        return tensor

    def _arr2rgb(self, arr):
        """
        Args:
            arr: ndarray
        return tensor(RGB)
        """
        arr = arr / (2**8-1)
        arr = arr.transpose((2, 0, 1))
        tensor = torch.from_numpy(arr).float()
        return tensor

    def _arr2ab(self, arr):
        """
        Args:
            arr: ndarray
        return tensor(LAB)
        """
        arr = rgb2lab(arr)[:,:,1:]
        arr = (arr + 128) / 255
        arr = arr.transpose((2, 0, 1))
        tensor = torch.from_numpy(arr).float()
        return tensor

    def _arr2lab(self, arr):
        """
        Args:
            arr: ndarray
        return tensor(LAB)
        """
        arr = rgb2lab(arr)
        arr[:,:,:1] = arr[:,:,:1] / 100
        arr[:,:,1:] = (arr[:,:,1:] + 128) / 255
        arr = arr.transpose((2, 0, 1))
        tensor = torch.from_numpy(arr).float()
        return tensor



class G2RGB(Basic):
    def __init__(
            self, root, split='all', transform=None):
        """
        8bit RGB information.
        Args:
            root (str): root of dataset
            split (str): part of the dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root, split, transform)

        self.src_ch = 1
        self.tar_ch = 3

    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = Image.open(src_file).convert('RGB')
        tar = Image.open(tar_file).convert('RGB')
        sample = {'src': src, 'tar': tar}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['src'] = np.array(sample['src'])
            sample['tar'] = np.array(sample['tar'])
        # src => arr => tensor(L)
        src = self._arr2gray(sample['src'])
        # tar => arr => tensor(RGB)
        tar = self._arr2rgb(sample['tar'])
        sample = {"src":src,
                  "tar":tar,
                  "idx":idx,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # convert bayer array to RGB
        src_rgb = self._rgb2img(src)
        tar_rgb = self._rgb2img(tar)
        if src_rgb.shape[:2] != tar_rgb.shape[:2]:
            src_rgb = cv2.resize(src_rgb, dsize=tar_rgb.shape[:2][::-1])
        vis_img = np.concatenate([src_rgb, tar_rgb], axis=1)
        save_dir = os.path.join(DIR, "./example", self.root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}.png".format(save_dir, self.split, idx), vis_img)



def load_dataset(root, mode="training"):
    """
    Args:
        root (str): root of dataset
        version (str): version of dataset
        mode (str): ['training', 'evaluating']
    """
    trainset = G2RGB(root=root, split="train")
    valset = G2RGB(root=root, split="val")
    testset = G2RGB(root=root, split="test")
    return trainset, valset, testset



if __name__ == "__main__":
    # setup parameters
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-idx', type=int, default=0,
                        help='index of sample image')
    parser.add_argument('-root', type=str, default='Sat2Aerx4',
                        help='root of the dataset')
    args = parser.parse_args()
    idx = args.idx
    root = args.root
    for stage in ["training", "testing"]:
        trainset, valset, testset = load_dataset(root, stage)
        # print("Load train set = {} examples, val set = {} examples".format(
        #     len(trainset), len(valset)))
        sample = trainset[idx]
        trainset.show(idx)
        valset.show(idx)
        testset.show(idx)
        print("Tensor size of {}/{}".format(root, stage))
        print("\tsrc:", sample["src"].shape,
                "tar:", sample["tar"].shape,)
