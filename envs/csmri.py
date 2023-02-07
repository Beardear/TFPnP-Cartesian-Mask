import sys
import json
import scipy.io as sio
import torch
import numpy as np
import argparse
import cv2
from utils import util
from utils.util import to_numpy, to_tensor
from data.noise_models import GaussianModelC, GaussianModelD
from utils.transforms import real2complex, complex2real, fft2
from envs.base import BaseEnv
from sigpy.mri.sim import birdcage_maps

BOOL = True if float(torch.__version__[:3]) >= 1.3 else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128


class CSMRI(BaseEnv):
    def __init__(self, train_loader, solver,
                 max_step=6, env_batch=64,
                 writer=None):
        # # ADMM_variables (3) + y0 (2) + ATy0 (1) + mask (1) + T (1) + sigma_n (1)
        super(CSMRI, self).__init__(train_loader, solver, max_step, env_batch, 6, writer) 
        self.keys = set(['y0', 'x0', 'ATy0', 'gt', 'mask', 'sigma_n'])
    
    def reset(self, test=False, data=None, loop_penalty=None):
        self.test = test
        self.stepnum = 0  # # of step in an episode
        N = self.env_batch if not test else 1

        if not test: # train mode
            try:
                data = self.data_iterator.next()
            except StopIteration:
                self.data_iterator = iter(self.train_loader)
                data = self.data_iterator.next()
        else:  # test mode
            assert data is not None

        for key in self.keys:
            data[key] = data[key].to(device=device)

        self.data = data
        self.variables = self.solver.reset(data)       # [x, z, u]
        self.idx_left = torch.arange(0, N).to(device)

        self.gt = self.data['gt']
        self.output = complex2real(self.data['ATy0'].clone().detach())
        self.input = complex2real(self.data['ATy0'].clone().detach())

        self.last_psnr = self.init_psnr = self.cal_psnr()

        self.loop_penalty = loop_penalty

        return self.observation()
    
    def observation(self):
        # (complex value) y0 (1) + ATy0 (1) + ADMM_variables (3) + mask (1) + T (1)
        idx_left = self.idx_left

        gt = real2complex(self.gt[idx_left, ...])
        y0 = self.data['y0'][idx_left, ...]
        ATy0 = self.data['ATy0'][idx_left, ...]
        variables = self.variables[idx_left, ...]
        mask = real2complex(self.data['mask'][idx_left, ...].unsqueeze(1)).float()
        sigma_n = self.data['sigma_n'][idx_left, ...]
        
        # _mask = self.data['mask'][idx_left, ...].unsqueeze(1)
        # _mask = _mask.bool() if BOOL else _mask.byte()

        # y0_ = fft2(gt) 
        # y0_[_mask, :] =  y0_[_mask, :]

        # res = y0_ - y0

        # sio.savemat('out.mat',{'out':res.detach().cpu().numpy()})
        # exit(0)

        N = y0.shape[0]
        
        T = (torch.ones([N, 1, width, width, 2], dtype=torch.float32) * self.stepnum / self.max_step).to(device)
        # ob = torch.cat([gt, variables, y0, ATy0, mask, T], 1)

        # + sigma_n 
        ob = torch.cat([gt, variables, y0, ATy0, mask, T, sigma_n], 1)
        
        # (complex value) + LoopPenalty (1)
        # LP = (torch.ones([N, 1, width, width, 2], dtype=torch.float32) * self.loop_penalty).to(device)
        # ob = torch.cat([gt, variables, y0, ATy0, mask, T, LP], 1)
        return ob

    def step(self, action, idx_stop, mask_ind):
        with torch.no_grad(): # no grad        
            idx_left = self.idx_left
                        
            _mask = self.data['mask'][idx_left, ...].unsqueeze(1)
            _mask = _mask.bool() if BOOL else _mask.byte()

            mask_ind = mask_ind.unsqueeze(1)
            mask_ind = mask_ind.unsqueeze(3)
            mask_ind = mask_ind.expand(-1,-1,-1, _mask.shape[-1])
            _mask = (_mask * mask_ind).bool()

            next_variables = self.solver(
                action, self.variables[idx_left, ...], 
                self.data['y0'][idx_left, ...], 
                _mask,
                torch.ones(idx_left.shape[0]) * self.stepnum
            )

            # use prior idx_left to update output
            self.output[idx_left, ...] = complex2real(next_variables[:,0:1,...])  # real x
            self.variables[idx_left,...] = next_variables
            self.stepnum += 1

            updated_ob = self.observation()
            reward = self.cal_reward() 
            all_done = self.update_idx(idx_stop)  # update idx_left
            ob = self.observation()

        if self.stepnum == self.max_step:
            all_done = True
            done = torch.ones_like(idx_stop)
        else:
            done = idx_stop.detach()

        if not self.test:
            if all_done:
                if self.writer is not None:
                    self.writer.add_scalar('train/mean_psnr', to_numpy(self.cal_psnr().mean()), self.log)
                self.log += 1
        
        return ob.detach(), updated_ob.detach(), reward, done, all_done
