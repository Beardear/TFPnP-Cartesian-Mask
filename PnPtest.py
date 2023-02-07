import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io.matlab.mio import loadmat
import torch
import os
import numpy as np
import PIL.Image as Image

from torch import nn
from os.path import join
from collections import namedtuple
from utils import transforms
from utils.util import to_numpy
from PnP.denoiser import UNet
from PnP.solver import PnPSolver
from sigpy.mri.sim import birdcage_maps
from utils import util
from train_setup import TrainOptions, Trainer

import scipy.io as sio
from PnP.denoiser import UNet

from sigpy.mri.sim import birdcage_maps

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def c2r_(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros( inp.shape+(2,),dtype=dtype)
    out[...,0]=inp.real
    out[...,1]=inp.imag
    return out

#%%
def complex_conjdot_torch(x: torch.Tensor, y: torch.Tensor):
    x_real, x_imag = torch.unbind(x, -1)
    y_real, y_imag = torch.unbind(y, -1)
    y_imag = y_imag * (-1)
    x_ = x_real * y_real
    y_ = x_imag * y_imag
    # res_real = torch.mul(x_real, y_real) - torch.mul(x_imag, y_imag)
    # res_imag = torch.mul(x_real, y_imag) + torch.mul(x_imag, y_real)

    return torch.stack([x_, y_], -1)

def _pre_img(img):
    img = to_numpy(img[0,...])
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

#convert x to n*1 vector
def vec(x):
    return x.view(x.shape[0], -1, 1)

from numpy.lib.stride_tricks import as_strided
from numpy.fft import ifftshift

def __normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def __cartesian_mask(imgSize, fold, centred=False):
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


if __name__ == "__main__":

    mridir = './../Data/Data/Medical7_2020/radial_128_4/5'
    sigma = torch.tensor([5])
    gt = sio.loadmat(join(mridir,'Brain.mat'))['gt']
    gt = torch.from_numpy(gt).unsqueeze(0)
    y0 = sio.loadmat(join(mridir,'Brain.mat'))['y0']
    y0 = torch.from_numpy(y0).unsqueeze(0).to(device)
    mask = sio.loadmat(join(mridir,'Brain.mat'))['mask']
    mask = torch.from_numpy(mask).unsqueeze(0).to(device) 
    
    coils = birdcage_maps([64,128,128], r=1.5, nzz=8) # JM
    coils = c2r_(coils) # JM
    coils_ = np.zeros_like(coils)
    coils_[0,...] = np.ones([1,128,128,2])
    # coils_ = torch.from_numpy(coils_).type(torch.FloatTensor) # JM
    coils = torch.from_numpy(coils_).type(torch.FloatTensor).unsqueeze(0).to(device) # JM [64,128,128,2]

    
    
    gt_imag = torch.zeros_like(gt)
    gt_cplx = torch.stack([gt, gt_imag], -1)
    kx = transforms.fft2(gt_cplx)
    kx_noisy = kx + torch.randn(kx.shape) * (sigma/255)
    kx_noisy[:,~mask,:] = 0

    x0 = transforms.ifft2(kx_noisy).to(device) #gt_cplx.detach().clone().to(device)#
    z = x0.clone().detach().to(device)
    u = torch.zeros_like(x0).to(device)
    _mu = torch.tensor([0.15]).to(device)
    sigma_d = torch.tensor([0.0085]).to(device)
    option = TrainOptions()
    opt = option.parse()
    solver = PnPSolver(opt)

    x0_cpu = torch.absolute(torch.view_as_complex(x0)).to('cpu')
    ps_nr_ = util.torch_psnr(x0_cpu, gt)
    print(f'psnr of x0 is:{ps_nr_}')
    print('-------------')



    for i in range(100):
        
        h = transforms.complex2real(z - u)
        x = transforms.real2complex(solver.prox_fun(h, sigma_d))  # plug-and-play proximal mapping

        coilimg = coils * (x+u)
        z = transforms.fft2(coilimg)

        # z = transforms.fft2(x+u)



        temp = ((_mu * z.clone()) + y0) / (1 + _mu)

        mask_ = mask.unsqueeze(-1)
        z =  temp * mask_

        z = transforms.ifft2(z)
        # z[...,1] = 0 #JM
        
        z=torch.sum(complex_conjdot_torch(z, coils),dim=1, keepdim=True)
        # u step
        u = u + x - z

        output = torch.absolute(torch.view_as_complex(x)).to('cpu')
        ps_nr = util.torch_psnr(output, gt)
        print(torch.mean(ps_nr))
        




