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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()

DataParallel = DataParallelWithCallback


class DDPG:
    def __init__(self, solver, actor, writer, opt, base_dim=6, verbose=True):
        if torch.cuda.device_count() > 1:
            self.data_parallel = True
        else:
            self.data_parallel = False

        self.solver = solver
        self.num_var = solver.num_var
        self.max_step = opt.max_step
        self.env_batch = opt.env_batch

        self.action_bundle = opt.action_pack
        self.loop_penalty = opt.loop_penalty
        self.lambda_e = opt.lambda_e

        state_dim = base_dim + self.num_var  # ADMM_variables (3) + y0 (2) + ATy0 (1) + mask (1) + T (1) + sigma_n (1)

        self.actor = actor
        self.critic = ResNet_wobn(state_dim, 18, 1) 
        self.critic_target = ResNet_wobn(state_dim, 18, 1)

        if verbose:
            print(self.actor)

        self.actor_optim  = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim  = Adam(self.critic.parameters(), lr=1e-2)

        resume = opt.resume
        resume_step = opt.resume_step

        if (resume != None):
            resume = join('./checkpoints', resume)
            print('[i] loading weight from {}... step: {}'.format(resume, resume_step))
            self.load_weights(resume, resume_step)

        # util.hard_update(self.actor_target, self.actor)
        util.hard_update(self.critic_target, self.critic)
        
        # Create replay buffer
        self.memory = rpm(opt.rmsize * opt.max_step)

        # Hyper-parameters
        self.tau = opt.tau  # target network update ratio
        self.discount = opt.discount

        # Tensorboard
        self.writer = writer
        self.log = 0
        
        self.state = [None] * self.env_batch    # Most recent state
        self.action = [None] * self.env_batch   # Most recent action

        self.choose_device()

    def update_policy(self, lr, mask_ind_temp):
        self.log += 1
        
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]
            
        # Sample batch
        ########### generate  state ##########
        state = self.memory.sample_batch(self.env_batch, device)

        action, action_logprob, idx_stop, dist_entropy, mask_ind_out = self.play(state)
        mask_ind_temp.append(mask_ind_out.cpu().detach().numpy())
        
        Q, Q_target, V_cur, one_reward = self.evaluate(state, action, idx_stop, mask_ind_out)
        advantage = (Q_target - V_cur).clone().detach()
        # advantage = torch.clamp(advantage, max=1)
        
        # policy_loss = - Q.mean() # fixed iteration
        # policy_loss = - (Q + action_logprob * advantage).mean()
        policy_loss = - (Q + action_logprob * advantage + self.lambda_e * dist_entropy).mean()

        value_loss = criterion(Q_target, V_cur)
        
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        # print(self.critic.fc.weight.grad)
        # print(self.actor.fc.weight.grad)
        self.critic_optim.step()

        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()
        
        # Target update
        # util.soft_update(self.actor_target, self.actor, self.tau)
        util.soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss.item(), value_loss.item(), dist_entropy.mean().item()

    def observe(self, reward, old_state, updated_state, done):
        s0 = old_state.clone().detach().cpu()

        for i in range(s0.shape[0]):
            self.memory.append(s0[i,...])

    def noise_action(self, noise_factor, state, action):        
        raise NotImplementedError
    
    def select_action(self, state, noise_factor=0, test=False):
        self.eval()

        with torch.no_grad(): # run without calculating gradient
            action, _, idx_stop, _, mask_ind_out = self.play(state, test=test)

        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)

        self.train()
        self.action = action #mu, sigma_d

        return action, idx_stop, mask_ind_out

    def reset(self, obs, factor):
        # self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)  # exploration noise

    def load_weights(self, path, step=None):
        if path is None: return
        if step is None:
            self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
            self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        else:
            self.actor.load_state_dict(torch.load('{}/actor_{:07d}.pkl'.format(path, step)))
            self.critic.load_state_dict(torch.load('{}/critic_{:07d}.pkl'.format(path, step)))
        
    def save_model(self, path, step=None):
        if self.data_parallel:
            self.actor = self.actor.module
            self.critic = self.critic.module       
            # self.actor_target = self.actor_target.module
            self.critic_target = self.critic_target.module      

        self.actor.cpu()
        self.critic.cpu()
        if step is None:
            torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(path))
            torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(path))
        else:
            torch.save(self.actor.state_dict(),'{}/actor_{:07d}.pkl'.format(path, step))
            torch.save(self.critic.state_dict(),'{}/critic_{:07d}.pkl'.format(path, step))    
        
        self.choose_device()

    def eval(self):
        self.actor.eval()
        # self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def train(self):
        self.actor.train()
        # self.actor_target.train()
        self.critic.train()
        self.critic_target.train()
    
    def choose_device(self):
        self.actor.to(device)
        # self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

        if self.data_parallel:
            self.actor = DataParallel(self.actor)
            # self.actor_target = DataParallel(self.actor_target)
            self.critic = DataParallel(self.critic)
            self.critic_target = DataParallel(self.critic_target)
