import numpy as np
import scipy.stats as stats
from os.path import join
import torch


class GaussianModelC:  # continuous noise levels
    def __init__(self, low_sigma=0, high_sigma=55):
        super().__init__()
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        
    def __call__(self, x, **kwargs):
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        
        # y = x + np.random.randn(*x.shape).astype(np.float32) * sigma
        sigma = sigma / 255.
        y = x + torch.randn(*x.shape) * sigma
        sigma = torch.ones_like(y) * sigma

        # return y.astype(np.float32), np.float32(sigma)
        return y, sigma


class GaussianModelD:  # discrete noise levels
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas
        
    def __call__(self, x, idx=None, **kwargs):
        if idx is not None:
            sigma = self.sigmas[idx]
        else:
            sigma = np.random.choice(self.sigmas)

        sigma = sigma / 255.
        # y = x + np.random.randn(*x.shape).astype(np.float32) * sigma
        y = x + torch.randn(*x.shape) * sigma
        
        sigma = torch.ones_like(y) * sigma
            
        # return y.astype(np.float32), np.float32(sigma)
        return y, sigma


class GaussianModelP:  # noise percentages
    def __init__(self, sigmas_p, batch_mode=False):
        super().__init__()
        self.sigmas_p = sigmas_p
        self.batch_mode = batch_mode
        
    def __call__(self, x, **kwargs):
        if not self.batch_mode:
            sigma = np.random.choice(self.sigmas_p)
            y = x + torch.randn_like(x) * torch.mean(torch.abs(x)) * sigma
            # sigma = torch.ones_like(y) * sigma
        else:
            N = x.shape[0]
            sigma = np.random.choice(self.sigmas_p, size=N)
            sigma = torch.from_numpy(sigma).view(N, 1, 1, 1).float().to(x.device)
            x_mean = torch.mean(torch.abs(x).view(N, -1), dim=1).view(N, 1, 1, 1)

            y = x + torch.randn_like(x) * x_mean * sigma
            # sigma = torch.ones_like(y) * sigma            

        return y.float(), sigma
