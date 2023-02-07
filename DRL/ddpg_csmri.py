import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from utils import util
from utils.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from utils.transforms import complex2channel, complex2real, real2complex
from os.path import join
from DRL.ddpg import DDPG


BOOL = True if float(torch.__version__[:3]) >= 1.3 else False


class DDPG_CSMRI(DDPG):
    def __init__(self, solver, actor, writer, opt, base_dim=6, verbose=True):
        super(DDPG_CSMRI, self).__init__(solver, actor, writer, opt, base_dim, verbose)
        self.cal_reward = util.torch_psnr

    def play(self, state, target=False, test=False, idx_stop=None, mask_ind=None):
        # ob: torch.cat([gt, variables, y0, ATy0, mask, T, sigma_n], 1) -> torch.cat([variables, y0, ATy0, mask, T, sigma_n], 1)
        # complex -> real
        assert state.ndimension() == 5
        num_var = self.num_var #num_var=3
        state = torch.cat([
            complex2real(state[:, 1:num_var+1, ...]), 
            complex2channel(state[:, num_var+1:num_var+2, ...]), 
            complex2real(state[:, num_var+2:, ...]),
        ], 1)
        
        if target:
            action, action_logprob, idx_stop, dist_entropy = self.actor_target(state, idx_stop, not test)
        else:
            action, action_logprob, idx_stop, dist_entropy, mask_ind_out = self.actor(state, idx_stop, not test, mask_ind)

        return action, action_logprob, idx_stop, dist_entropy, mask_ind_out
        
    def evaluate(self, state, action, idx_stop, mask_ind):  # only perform critic (no actor)
        num_var = self.num_var
        N = state.shape[0]
        
        gt = state[:, 0:1, ...]
        variables = state[:, 1:num_var+1, ...]
        y0 = state[:, num_var+1:num_var+2, ...]
        ATy0 = state[:, num_var+2:num_var+3, ...]
        mask = state[:, num_var+3:num_var+4, ...]
        T = state[:, num_var+4:num_var+5, ...]
        sigma_n = state[:, num_var+5:num_var+6, ...]


        stepnum = (complex2real(T) * self.max_step).view(N, -1).mean(dim=1)
    
        _mask = complex2real(mask).bool() if BOOL else complex2real(mask).byte()
        #-----------------mid -16 lines = 1, else=0---------------
        
        # mask_ind[:, int(mask_ind.shape[1]/2-8):int(mask_ind.shape[1]/2+8)] = 1
        mask_ind = mask_ind.unsqueeze(1)
        mask_ind = mask_ind.unsqueeze(3)
        mask_ind = mask_ind.expand(-1,-1,-1, _mask.shape[-1])
        _mask = (_mask * mask_ind).bool()
        

        next_variables = self.solver(
            action, variables,
            y0, _mask,
            stepnum
        )

        gt_real = complex2real(gt)
        output0 = complex2real(variables[:, 0:1,...])
        output1 = complex2real(next_variables[:, 0:1,...])
        
        one_reward = self.cal_reward(output1, gt_real) - self.cal_reward(output0, gt_real) - self.loop_penalty
        # one_reward = cal_reward(output1, gt_real) - cal_reward(output0, gt_real) - (self.loop_penalty * (~idx_stop.byte()).float()).unsqueeze(1)
        
        # ADMM_variables0 (3) + ADMM_variables1 (3) + y0 (2) + ATy0 (1) + mask (1) + T (1)
        cur_eval_state = torch.cat([
            complex2real(variables),
            complex2channel(y0),
            complex2real(ATy0),
            complex2real(mask),
            complex2real(T),
            complex2real(sigma_n),
        ], 1)

        next_eval_state = torch.cat([
            complex2real(next_variables),
            complex2channel(y0),
            complex2real(ATy0),
            complex2real(mask),
            complex2real(T) + 1/self.max_step,
            complex2real(sigma_n),
        ], 1)
        
        V_next = self.critic_target(next_eval_state) 
        V_next = self.discount * ((1 - idx_stop.float()).view(-1, 1)) * V_next
        # Q = one_reward
        Q = V_next + one_reward

        # advantage: temporal diff
        V_cur = self.critic(cur_eval_state)

        with torch.no_grad():
            V_next_target = self.critic_target(next_eval_state)
            V_next_target = self.discount * ((1 - idx_stop.float()).view(-1, 1)) * V_next_target
            Q_target = V_next_target + one_reward.detach()

        if self.log % 20 == 0 and self.writer is not None:
            self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
            self.writer.add_scalar('train/one_reward', one_reward.mean(), self.log)

        # import ipdb; ipdb.set_trace()
        return Q, Q_target, V_cur, one_reward
