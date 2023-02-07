import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import random
import PIL.Image as Image
from utils.util import to_numpy
from scipy.io import savemat, loadmat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(data, policy, env, loop_penalty=0.05, max_step=6):
    observation = None

    assert data['gt'].shape[0] == 1
    # reset at the start of episode                
    observation = env.reset(test=True, data=data, loop_penalty=loop_penalty)

    # input, _, gt = env.get_images()
    # Image.fromarray(input).save('./scheme/input.png')
    # Image.fromarray(gt).save('./scheme/gt.png')

    episode_steps = 0
    episode_reward = 0.
    assert observation is not None
    # start episode
    episode_reward = np.zeros(1)            
    all_done = False

    sigma_d_seq = []
    mu_seq = []
    psnr_seq = [env.cal_psnr().item()]
    reward_seq = [0]

    while (episode_steps < max_step or not max_step):
        
        action, idx_stop = policy(observation, test=True)
        observation, updated_observation, reward, done, all_done = env.step(action, idx_stop)
        
        if not all_done: 
            reward = reward - loop_penalty

        episode_reward += reward.item()
        episode_steps += 1

        half = action.shape[1] // 2
        
        sigma_d_seq.extend(list(to_numpy(action[0, :half]*255)))
        mu_seq.extend(list(to_numpy(action[0, half:])))

        cur_psnr = env.cal_psnr()
        psnr_seq.append(cur_psnr.item())      
        reward_seq.append(reward.item())

        # _, output, _ = env.get_images()
        # Image.fromarray(output).save('./scheme/{}.png'.format(episode_steps))

        if all_done:
            break

    input, output, gt = env.get_images()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning-free Plug-and-Play Proximal Algorithm')

    # hyper-parameter
    parser.add_argument('--exp', default='csmri_admm_5x6_48', type=str, help='name of experiment')
    parser.add_argument('--warmup', default=100, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='discount factor')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=48, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=6, type=int, help='max length for episode')
    parser.add_argument('--resume', '-r', default='csmri_admm_5x6_48', type=str, help='Resuming model path')
    parser.add_argument('--resume_step', '-rs', default=15000, type=int, help='Resuming model step')
    parser.add_argument('--output', default='./checkpoints', type=str, help='resuming model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--loop_penalty', '-lp', type=float, default=0.05, help='penalty of loop')        
    parser.add_argument('--action_pack', '-ap', type=int, default=5, help='pack of action')
    parser.add_argument('--lambda_e', '-le', type=float, default=0, help='penalty of loop')        
    parser.add_argument('--denoiser', type=str, default='unet', help='denoising network')        
    parser.add_argument('--solver', type=str, default='admm', help='invoked solver')

    opt = parser.parse_args()

    if opt.resume is None and opt.resume_step is not None:
        opt.resume = opt.exp

    import PnP, DRL
    from DRL import DDPG_CSMRI
    from DRL.evaluator import Evaluator
    from envs.csmri import CSMRI
    from data.datasets import CSMRIDataset, CSMRIEvalDataset
    from data.noise_models import GaussianModelD
    import utils.util as util

    import socket
    from os.path import join
    from scipy.io import loadmat

    writer = None

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
    if torch.cuda.device_count() > 1:
        print("[i] Use", torch.cuda.device_count(), "GPUs...")

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # modify to your own path
    # mridir = '/media/kaixuan/DATA/Papers/Code/Data/MRI'
    mridir = './../Data/Data'
        
    sampling_masks = ['radial_128_2']#, 'radial_128_4', 'radial_128_8']  # different masks
    # sampling_masks = ['radial_128_8']  # different masks
    # sigma_ns = [5]
    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)

    obs_masks = [loadmat(join(mridir, 'masks', '{}.mat'.format(sampling_mask))).get('mask') for sampling_mask in sampling_masks] 

    # train_dataset would not be used in testing mode, though it's necessary to be defined.
    train_dataset = CSMRIEvalDataset(join(mridir, 'Medical7_2020', sampling_masks[0], str(15)), fns=None)
    
    names = []
    names += ['m7x2_sigma15', 'm7x4_sigma15', 'm7x8_sigma15'] 


    val_datasets = [CSMRIEvalDataset(join(mridir, 'Medical7_2020', sampling_mask, str(15)), fns=None) for sampling_mask in sampling_masks]

    num_workers = 0 if opt.debug else 4

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.env_batch, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) for val_dataset in val_datasets]

    solver = PnP.get_solver(opt)
    actor = DRL.get_actor(opt)

    fenv = CSMRI(train_loader, solver, opt.max_step, opt.env_batch, writer=writer)

    agent = DDPG_CSMRI(solver, actor, writer, opt)

    evaluate = Evaluator(opt, val_loaders, names, writer)

    print('observation_space', fenv.observation_space)

    evaluate(fenv, agent.select_action, opt.resume_step, opt.loop_penalty)
    
