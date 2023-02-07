import cv2
import random
import numpy as np
import torch
import argparse
from utils import util
import time
import os
import scipy.io as sio



class Trainer:
    def __init__(self, opt, agent, env, evaluate, writer):
        self.opt = opt
        self.agent = agent
        self.env = env
        self.evaluate = evaluate
        self.writer = writer

    def train(self, lr_scheduler):
        opt = self.opt
        agent = self.agent
        env = self.env
        evaluate = self.evaluate
        step = self.opt.resume_step
        writer = self.writer

        train_times = opt.train_times
        env_batch = opt.env_batch
        validate_interval = opt.validate_interval
        max_step = opt.max_step
        debug = opt.debug
        episode_train_times = opt.episode_train_times
        resume = opt.resume
        output = opt.output
        time_stamp = time.time()
        episode = episode_steps = 0
        step = 0 if step is None else step
        observation = None
        all_done = False
        noise_factor = opt.noise_factor

        while step <= train_times:
            step += 1
            episode_steps += 1
            # reset if it is the start of episode
            if observation is None:
                observation = env.reset()
                agent.reset(observation, noise_factor)
            action, idx_stop, mask_ind_out = agent.select_action(observation, noise_factor=noise_factor)
            old_observation = observation
            observation, updated_observation, reward, done, all_done = env.step(action, idx_stop, mask_ind_out)
            agent.observe(reward, old_observation, updated_observation, done)

            if (episode_steps >= max_step and max_step) or all_done:
                if step > opt.warmup:
                    # [optional] evaluate
                    if evaluate is not None:
                        if (episode > 0 and validate_interval > 0 and episode % validate_interval == 0):
                            evaluate(env, agent.select_action, step, opt.loop_penalty)
                            agent.save_model(output)

                train_time_interval = time.time() - time_stamp
                time_stamp = time.time()

                tot_Q = 0.
                tot_value_loss = 0.
                tot_dist_entropy = 0.
                mean_Q = 0.
                mean_dist_entropy = 0.
                mean_value_loss = 0.

                if step > opt.warmup:
                    # if step < 10000:
                    #     lr = (3e-4, 1e-3)
                    # else:
                    #     lr = (1e-4, 3e-4)
                    lr = lr_scheduler(step)
                    mask_ind_temp = []
                    for i in range(episode_train_times):
                        Q, value_loss, dist_entropy = agent.update_policy(lr, mask_ind_temp)
                        tot_Q += Q
                        tot_value_loss += value_loss
                        tot_dist_entropy += dist_entropy
                    
                    mdic = {'mask_ind_temp':mask_ind_temp}
                    # sio.savemat(f'mask_ind_temp{step}.mat', mdic)

                    mean_Q = tot_Q / episode_train_times
                    mean_dist_entropy = tot_dist_entropy / episode_train_times
                    mean_value_loss = tot_value_loss / episode_train_times
                    if writer is not None:
                        writer.add_scalar('train/critic_lr', lr[0], step)
                        writer.add_scalar('train/actor_lr', lr[1], step)
                        writer.add_scalar('train/Q', mean_Q, step)
                        writer.add_scalar('train/dist_entropy', mean_dist_entropy, step)
                        writer.add_scalar('train/critic_loss', mean_value_loss, step)

                util.prBlack('#{}: steps: {} | interval_time: {:.2f} | train_time: {:.2f} | Q: {:.2f} | dist_entropy: {:.2f} | critic_loss: {:.2f}' \
                    .format(episode, step, train_time_interval, time.time()-time_stamp, mean_Q, mean_dist_entropy, mean_value_loss))

                time_stamp = time.time()
                # reset
                observation = None
                episode_steps = 0
                episode += 1
                all_done = False

            if step % 100 == 0:
                sio.savemat(f'mask_ind_temp{step}.mat', mdic)
                if evaluate is not None:
                    evaluate(env, agent.select_action, step, opt.loop_penalty)

                util.prRed('Saving model at Step_{:07d}...'.format(step))
                agent.save_model(output, step)

                # reset
                observation = None
                episode_steps = 0
                episode += 1
                all_done = False


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Tuning-free Plug-and-Play Proximal Algorithm')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp', default='csmri_admm_5x6_48', type=str, help='name of experiment')
        self.parser.add_argument('--warmup', default=20, type=int, help='timestep without training but only filling the replay memory')
        self.parser.add_argument('--discount', default=0.99, type=float, help='discount factor')
        self.parser.add_argument('--rmsize', default=480, type=int, help='replay memory size')
        self.parser.add_argument('--env_batch', default=20, type=int, help='concurrent environment number')
        self.parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
        self.parser.add_argument('--max_step', default=9, type=int, help='max length for episode')
        self.parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise') # 0.04
        self.parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
        self.parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
        self.parser.add_argument('--train_times', default=15000, type=int, help='total traintimes')
        self.parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
        self.parser.add_argument('--resume', '-r', default=None, type=str, help='Resuming model path')
        self.parser.add_argument('--resume_step', '-rs', default=None, type=int, help='Resuming model step')
        self.parser.add_argument('--output', default='./checkpoints', type=str, help='resuming model path for testing')
        self.parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
        self.parser.add_argument('--seed', default=1234, type=int, help='random seed')
        self.parser.add_argument('--loop_penalty', '-lp', type=float, default=0.05, help='penalty of loop')        
        self.parser.add_argument('--action_pack', '-ap', type=int, default=5, help='pack of action')
        self.parser.add_argument('--lambda_e', '-le', type=float, default=0.2, help='penalty of loop')
        self.parser.add_argument('--denoiser', type=str, default='unet', help='denoising network')
        self.parser.add_argument('--solver', type=str, default='admm', help='invoked solver')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()
        opt.output = util.get_output_folder(opt.output, opt.exp)
        print('[i] Exp dir: {}'.format(opt.output))

        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)

        if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
        if torch.cuda.device_count() > 1:
            print("[i] Use", torch.cuda.device_count(), "GPUs...")

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        self.opt = opt

        return opt
