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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
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


class ADMMSolver_CSMRI(PnPSolver):
    def __init__(self, opt):
        super(ADMMSolver_CSMRI, self).__init__(opt)        
        self.num_var = 3
        # self.coils = np.expand_dims(birdcage_maps([64,128,128], r=1.5, nzz=8),0)
        # self.coils = c2r_(self.coils)
        # self.coils = torch.from_numpy(self.coils).type(torch.FloatTensor)

    def reset(self, data):
        x = data['x0'].clone().detach()
        z = x.clone().detach()
        u = torch.zeros_like(x)
        variables = torch.cat([x, z, u], dim=1)
        return variables

    def forward(self, action, variables, y0, mask, stepnum):
        # state: torch.cat([x, z, u], dim=1)
        assert variables.ndimension() == 5 and variables.shape[1] == 3
        
        # decode action
        sigma_d = action['sigma_d']
        mu = action['mu']
        
        num_loops = self.num_loops
        N = variables.shape[0]

        x, z, u = torch.split(variables, 1, dim=1)


       # mask_ind full[:,mask_ind==0] = 0
        for i in range(num_loops):
            # img_np = _pre_img(transforms.complex2real(x))
            # k = 0
            # while True:
            #     if not os.path.exists('./scheme/{}.png'.format(k)):
            #         Image.fromarray(img_np[0,...]).save('./scheme/{}.png'.format(k)) 
            #         break
            #     k += 1    

            # x step
            x = transforms.real2complex(self.prox_fun(transforms.complex2real(z - u), sigma_d[:, i]))  # plug-and-play proximal mapping

            # z step
            # z = torch.fft(x + u, 2)
            # coilimg = coils * (x+u)
            z = transforms.fft2(x + u)#z = transforms.fft2(coilimg) #
            # z[mask, :] = y0[mask, :]  # for the case of mu = 0
            # z[mask, :] = ((mu[:, i].view(N, 1, 1) * z[mask, :].view(N, -1, 2) + y0[mask, :].view(N, -1, 2)) / (1 + mu[:, i]).view(N, 1, 1)).view(-1, 2)            

            _mu = mu[:, i].view(N, 1, 1, 1, 1)
            temp = ((_mu * z.clone()) + y0) / (1 + _mu)
            
            z[mask, :] =  temp[mask, :]

            z = transforms.ifft2(z)
            z[...,1] = 0 
            # z=torch.sum(complex_conjdot_torch(z, coils),dim=1)

            # u step
            u = u + x - z
        
        next_variables = torch.cat([x, z, u], dim=1)

        return next_variables

# %%
