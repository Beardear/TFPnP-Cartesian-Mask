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


BOOL = True if float(torch.__version__[:3]) >= 1.3 else False

BaseDataset = torchdata.Dataset


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(imgSize, fold, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(imgSize[:-2])), imgSize[-2], imgSize[-1]
    pdf_x = __normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*fold)
    n_lines  = int(Nx / fold)
    sample_n = int(n_lines / 3)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(imgSize)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    mask = mask.astype(np.float32)

    re = torch.from_numpy(mask)
    re = torch.stack([re, re], -1)
    re = re.unsqueeze_(0)

    return re


class ImageFolder(BaseDataset):
    def __init__(self, datadir, fns=None, target_size=None, repeat=1):
        super(ImageFolder, self).__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]        
        self.target_size = target_size
        self.repeat = repeat

    def __getitem__(self, index):
        index = index % len(self.fns)        
        imgpath = join(self.datadir, self.fns[index])        
        name = os.path.splitext(self.fns[index])[0]
        
        target = Image.open(imgpath).convert('L')

        if self.target_size is not None:
            ow, oh = target.size
            target = scale_height(target, self.target_size) if ow >= oh else scale_width(target, self.target_size)

        target = np.array(target, dtype=np.float32) / 255.0        

        if target.ndim == 2:
            target = target[None]
        elif target.ndim == 3:
            target = target.transpose((2,0,1))
        else:
            raise NotImplementedError
        
        return {'gt': target, 'name': name}

    def __len__(self):
        return len(self.fns) * self.repeat


class CSMRIDataset(BaseDataset):
    def __init__(self, datadir, fns, masks, noise_model=None, size=None, target_size=None, repeat=1):
        super(CSMRIDataset, self).__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]      
        self.masks = masks
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size
        self.reset()

    def __getitem__(self, index):
        mask = self.masks[np.random.randint(0, len(self.masks))]
        
        # mask = torch.squeeze(cartesian_mask(self.masks[0].shape, random.choice([2,4,8])))
        # mask = torch.squeeze(cartesian_mask(self.masks[0].shape, 2))
        # mask = torch.absolute(torch.view_as_complex(mask)).detach().numpy()
        
        # mask = sio.loadmat('fixed_cartesian_2.mat')['mask']

        if BOOL:
            mask = mask.astype(np.bool)
        
        sigma_n = 0

        index = index % len(self.fns)
        imgpath = join(self.datadir, self.fns[index])
        target = Image.open(imgpath).convert('L')

        if self.target_size is not None:
            ow, oh = target.size            
            target = scale_height(target, self.target_size) if ow >= oh else scale_width(target, self.target_size)        

        target = np.array(target, dtype=np.float32) / 255.0

        if target.ndim == 2:
            target = target[None]
        elif target.ndim == 3:
            target = target.transpose((2,0,1))
        else:
            raise NotImplementedError

        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)
        
        y0 = transforms.fft2(torch.stack([target, torch.zeros_like(target)], dim=-1))
        # y0[:, ~mask, :] = 0

        if self.noise_model is not None:
            y0, sigma_n = self.noise_model(y0)

        y0[:, ~mask, :] = 0


        ATy0 = transforms.ifft2(y0)
        ATy0[...,1] = 0
        x0 = ATy0.clone().detach()

        dic = {'y0': y0, 'x0': x0, 'ATy0': ATy0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n}

        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


class CSMRIEvalDataset(BaseDataset):
    def __init__(self, datadir, fns=None):
        super(CSMRIEvalDataset, self).__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".mat")]      
        self.reset()         
    
    def __getitem__(self, index):
        fn = self.fns[index]
        mat = loadmat(join(self.datadir, fn))
        mat['name'] = mat['name'].item()
        mat.pop('__globals__', None)
        mat.pop('__header__', None)
        mat.pop('__version__', None)
        # mat['mask'] = sio.loadmat('fixed_cartesian_2.mat')['mask']
        return mat
        
    def __len__(self):
        return len(self.fns)
