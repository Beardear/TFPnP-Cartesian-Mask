import torch
import numpy as np
from utils import util
from utils.util import to_numpy, to_tensor

width = 128

class BaseEnv:    
    def __init__(self, train_loader, solver,
                 max_step=6, env_batch=64, base_dim=6,
                 writer=None):
        self.train_loader = train_loader
        self.data_iterator = iter(self.train_loader)
        self.solver = solver
        self.writer = writer
        self.test = False
        
        self.env_batch = env_batch
        self.max_step = max_step  # max_episode_length
        self.observation_space = (self.env_batch, width, width, base_dim+solver.num_var) 
        self.log = 0

    def cal_psnr(self):
        return util.torch_psnr(self.output, self.gt)

    def cal_reward(self):
        psnr = self.cal_psnr()
        reward = (psnr - self.last_psnr)

        self.last_psnr = psnr
        return reward

    def update_idx(self, idx_stop):
        idx = idx_stop == 0  # left
        idx_left = self.idx_left
        self.idx_left = idx_left[idx]  # update idx_left
        
        if len(self.idx_left) == 0:
            all_done = True
        else:
            all_done = False
            
        return all_done

    # def update_idx(self, idx_stop):
    #     if self.test:
    #         idx = idx_stop == 0  # left
    #         idx_left = self.idx_left        
    #         self.idx_left = idx_left[idx]  # update idx_left
            
    #         if len(self.idx_left) == 0:
    #             all_done = True
    #         else:
    #             all_done = False
    #     else:
    #         all_done = False

    #     return all_done

    # def update_idx(self, idx_stop):  # disable early stopping behavior
    #     all_done = False
    #     return all_done

    def get_images(self):
        def _pre_img(img):
            img = to_numpy(img[0,...])
            img = np.repeat((np.clip(img, 0, 1) * 255).astype(np.uint8), 3, axis=0)
            return img
            
        assert self.gt.shape[0] == 1  # invoked only in eval mode

        input = _pre_img(self.input)        
        output = _pre_img(self.output)
        gt = _pre_img(self.gt)

        return input, output, gt
