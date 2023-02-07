import torch
import torch.nn as nn
from PnP.denoiser import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PnPSolver(nn.Module):
    def __init__(self, opt):
        super(PnPSolver, self).__init__()        
        denoiser = opt.denoiser
        num_loops = opt.action_pack
        
        print('[i] plugged denoiser: {}'.format(denoiser))
        if denoiser == 'unet':
            netG = UNet(2, 1).to(device)
            state_dict = torch.load('./unet-nm.pt', map_location=device)
        else:
            raise NotImplementedError

        netG.load_state_dict(state_dict)
        netG.eval()

        for param in netG.parameters():
            param.requires_grad = False
            
        self.netG = netG        
        self.num_loops = num_loops

    def prox_fun(self, x, sigma_d):
        N, C, H, W = x.shape

        if sigma_d.shape[0] == N:
            sigma_d = sigma_d.view(N, 1, 1, 1)
        
        noise_map = torch.ones(N, 1, H, W).to(x.device) * sigma_d
        out = self.netG(torch.cat([x, noise_map], dim=1))

        return torch.clamp(out, 0, 1)
