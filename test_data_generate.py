# dataset
import torch
import os
import glob
import numpy as np
import random
from os.path import join
import data.torchdata as torchdata
import h5py
import math
import torch.utils.data.sampler as sampler
from PIL import Image
import torch.utils.data as data
import sys
import cv2
from scipy.io import loadmat
from utils import transforms
from numpy.lib.stride_tricks import as_strided
from numpy.fft import ifftshift
import random
import scipy.io as sio
from data.noise_models import GaussianModelD

datadir = './../Data/Data/Medical_128'

_mask = sio.loadmat('fixed_cartesian_2.mat')['mask']

_mask = _mask.astype(np.bool)

noise_model = GaussianModelD([15])

sigma_n = 0
fns = [im for im in os.listdir(datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")] 

for index in range(len(fns)):
    imgpath = join(datadir, fns[index])
    # target = sio.loadmat(imgpath)['gt']
    target = Image.open(imgpath).convert('L')
        

    target = np.array(target, dtype=np.float32) / 255.0

    if target.ndim == 2:
        target = target[None]
    elif target.ndim == 3:
        target = target.transpose((2,0,1))
    else:
        raise NotImplementedError

    target = torch.from_numpy(target)
    mask = torch.from_numpy(_mask)

    y0 = transforms.fft2(torch.stack([target, torch.zeros_like(target)], dim=-1))
    # y0[:, ~mask, :] = 0

    if noise_model is not None:
        y0, sigma_n = noise_model(y0)

    y0[:, ~mask, :] = 0


    ATy0 = transforms.ifft2(y0)
    ATy0[...,1] = 0
    x0 = ATy0.clone().detach()

    dic = {'y0': y0.cpu().detach().numpy(), 'ATy0': ATy0.cpu().detach().numpy(), 'gt': target.cpu().detach().numpy(), 'x0': x0.cpu().detach().numpy(), 'sigma_n': sigma_n.cpu().detach().numpy(), 'mask': mask.cpu().detach().numpy()}
    # sio.savemat(f'gen{index}.mat', dic)


