#!/usr/bin/env python3
import cv2
import random
import numpy as np
import torch
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from utils import util
from utils.sync_batchnorm import DataParallelWithCallback
from tensorboardX import SummaryWriter
from DRL.evaluator import Evaluator
from train_setup import TrainOptions, Trainer



if __name__ == "__main__":
    import PnP, DRL
    from DRL import DDPG_CSMRI
    from envs.csmri import CSMRI
    # from PnP.solver_csmri import ADMMSolver_CSMRI    
    from data.datasets import CSMRIDataset, CSMRIEvalDataset
    from data.noise_models import GaussianModelD

    import socket
    from os.path import join
    from scipy.io import loadmat
    
    

    # traindir= '/media/kaixuan/DATA/Papers/Code/Data/Reflection/VOCdevkit/VOC2012/'
    # mridir = '/media/kaixuan/DATA/Papers/Code/Data/MRI'
    # testdir = '/media/kaixuan/DATA/Papers/Code/Data/testsets'
    
    
    mridir = './../Data/Data'
    testdir = './../Data/Data'

    option = TrainOptions()
    opt = option.parse()

    # writer = None
    writer = SummaryWriter('./train_log/{}'.format(opt.exp))
    # os.system('mkdir ./checkpoints/{}'.format(opt.exp))

    # sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']  # different masks
    sampling_masks = ['radial_128_2']  # different masks
    # sigma_ns = [5]
    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)    

    obs_masks = [loadmat(join(mridir, 'masks', '{}.mat'.format(sampling_mask))).get('mask') for sampling_mask in sampling_masks] 
    
    ## training with PASCAL VOC dataset (with image size 128x128)
    # train_dataset = CSMRIDataset(os.path.join(traindir, 'Images_128'), fns=None, masks=obs_masks, noise_model=noise_model)

    ## overfitting with test images as training dataset
    train_dataset = CSMRIDataset(join(testdir, 'Images_128'), fns=None, masks=obs_masks, noise_model=GaussianModelD([15]), repeat=12*100)
    
    # names = ['m7x2_sigma15', 'm7x4_sigma15', 'm7x8_sigma15']
    names = ['m7x2_sigma15']
    # val_datasets = [CSMRIEvalDataset(join(mridir, 'Medical7_2020', sampling_mask+'_Cartesian', str(15)), fns=None) for sampling_mask in sampling_masks]    
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

    if torch.cuda.device_count() > 1:
        # solver = util.DataParallel(solver)
        solver = DataParallelWithCallback(solver)

    fenv = CSMRI(train_loader, solver, opt.max_step, opt.env_batch, writer)

    agent = DDPG_CSMRI(solver, actor, writer, opt)

    evaluate = Evaluator(opt, val_loaders, names, writer)
    # evaluate = None
    print('observation_space', fenv.observation_space)

    def lr_scheduler(step):
        if step < 10000:
            lr = (3e-4, 1e-3)
        else:
            lr = (1e-4, 3e-4)
        return lr

    trainer = Trainer(opt, agent, fenv, evaluate, writer)
    trainer.train(lr_scheduler)
    
