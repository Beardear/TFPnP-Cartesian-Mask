import numpy as np
from utils.util import *
from utils import transforms
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import PIL.Image as Image
import os
from os.path import join

from scipy import io


class Evaluator(object):
    def __init__(self, opt, val_loaders, names, writer):  
        self.val_loaders = val_loaders
        self.names = names
        self.max_step = opt.max_step
        self.env_batch = opt.env_batch
        self.writer = writer

    def __call__(self, env, policy, step, loop_penalty=0):        
        torch.manual_seed(1111)  # fix
        basedir = 'suppl'
        cnt = 0
        for name, val_loader in zip(self.names, self.val_loaders):
            avg_meters = AverageMeters()        
            observation = None

            for k, data in enumerate(val_loader):
                cnt = cnt + 1
                assert data['gt'].shape[0] == 1
                # reset at the start of episode                
                observation = env.reset(test=True, data=data, loop_penalty=loop_penalty)
                
                input, _, gt = env.get_images()
                if not os.path.exists(join(basedir, name, str(k))):
                    os.makedirs(join(basedir, name, str(k)))

                # y0 = to_numpy(transforms.complex_abs(data['y0']))[0,0,...]
                # y0 = np.log(y0)
                # plt.imshow(y0, cmap='gray')
                # plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
                
                # plt.savefig('./{}/{}/FFT.png'.format(k), bbox_inches='tight', pad_inches=0)        
                # plt.clf()
                                           
                # Image.fromarray(input[0,...]).save(join(basedir, name, str(k), 'input.png'))
                # Image.fromarray(gt[0,...]).save(join(basedir, name, str(k), 'gt.png'))

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

                while (episode_steps < self.max_step or not self.max_step):
                    action, idx_stop, mask_ind_out = policy(observation, test=True)
                    observation, updated_observation, reward, done, all_done = env.step(action, idx_stop, mask_ind_out)
                    
                    if not all_done: 
                        reward = reward - loop_penalty

                    episode_reward += reward.item()
                    episode_steps += 1

                    sigma_d_seq.extend(list(to_numpy(action['sigma_d'][0]*255)))
                    # mu_seq.extend(list(to_numpy(action['mu'][0])))

                    cur_psnr = env.cal_psnr()
                    psnr_seq.append(cur_psnr.item())      
                    reward_seq.append(reward.item())
                    
                    _, output, _ = env.get_images()

                    # Image.fromarray(output[0,...]).save('k{}-episode_steps{}-{}.png'.format(k, episode_steps,cnt))
                    
                    
                    # _, output, _ = env.get_images()
                    # Image.fromarray(output[0,...]).save('./{}+{}.png'.format(k, episode_steps))

                    if all_done:
                        break

                input, output, gt = env.get_images()
                # io.savemat(name+'_output.mat', {'output':output})
                Image.fromarray(output[0,...]).save('./eval_results/k{}--{}.png'.format(k,cnt))

                # fig, _ = self.seq_plot(psnr_seq, 'Number of iterations (#IT.)', 'PSNR (dB)')
                # fig, _ = self.seq_plot(sigma_d_seq, 'Number of iterations (#IT.)', r'Denoising strength $\sigma$', 'blue')
                # plt.savefig(join(basedir, name, str(k), 'sigma.pdf'), bbox_inches='tight')
                # plt.clf()
                # fig, _ = self.seq_plot(mu_seq, 'Number of iterations (#IT.)', r'Penalty parameter $\mu$', 'orange')
                # plt.savefig(join(basedir, name, str(k), 'mu.pdf'), bbox_inches='tight')
                # plt.clf()
                
                if self.writer is not None:                    
                    fig, _ = self.seq_plot(psnr_seq, '#IT.', 'PSNR')
                    self.writer.add_figure('{}/lp{:.2f}/{}/psnr_seq'.format(name, loop_penalty, k), fig, step)
                    fig, _ = self.seq_plot(sigma_d_seq, '#IT.', r'$\sigma$')
                    self.writer.add_figure('{}/lp{:.2f}/{}/sigma_d_seq'.format(name, loop_penalty, k), fig, step)
                    
                    # fig, _ = self.seq_plot(mu_seq, '#IT.', r'$\mu$')
                    # self.writer.add_figure('{}/lp{:.2f}/{}/mu_seq'.format(name, loop_penalty, k), fig, step)                    
                    fig, ax = self.seq_plot(reward_seq, '#IT.', 'Reward')
                    ax.hlines(y=0, xmin=0, xmax=len(reward_seq)-1, linestyles='dotted', colors='r')
                    self.writer.add_figure('{}/lp{:.2f}/{}/reward_seq'.format(name, loop_penalty, k), fig, step)

                    self.writer.add_image('{}/lp{:.2f}/{}/_gt.png'.format(name, loop_penalty, k), gt, step)
                    self.writer.add_image('{}/lp{:.2f}/{}/_input.png'.format(name, loop_penalty, k), input, step)
                    self.writer.add_image('{}/lp{:.2f}/{}/_output.png'.format(name, loop_penalty, k), output, step)

                psnr_finished = env.cal_psnr()

                avg_meters.update({'acc_reward': episode_reward, 'psnr': psnr_finished, 'iters': episode_steps})
            prRed('Step_{:07d}: {} | loop_penalty: {:.2f} | {}'.format(step - 1, name, loop_penalty, avg_meters))

            if self.writer is not None:
                print('------------------------saved---------------------------')
                self.writer.add_scalar('validate/{}/lp{:.2f}/mean_acc_reward'.format(name, loop_penalty), avg_meters['acc_reward'], step)
                self.writer.add_scalar('validate/{}/lp{:.2f}/mean_psnr'.format(name, loop_penalty), avg_meters['psnr'], step)
                self.writer.add_scalar('validate/{}/lp{:.2f}/mean_iters'.format(name, loop_penalty), avg_meters['iters'], step)

            # return avg_meters

    def seq_plot(self, seq, xlabel, ylabel, color='blue'):        
        # fig, ax = plt.subplots(1, 1)
        # fig, ax = plt.subplots(1, 1, figsize=(6,4))
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        # ax.plot(np.array(seq))
        ax.plot(np.arange(1, len(seq)+1), np.array(seq), 'o--', markersize=10, linewidth=2, color=color)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        # plt.xticks(fontsize=16)
        xticks = list(range(1, len(seq)+1, max(len(seq)//5,1)))
        if xticks[-1] != len(seq):
            xticks.append(len(seq))

        plt.xticks(xticks, fontsize=16)
        
        return fig, ax
