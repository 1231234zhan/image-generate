import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm
import argparse
import pickle
import json
import random
import math
import time
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import cv2

_project_folder_ = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)
os.chdir(_project_folder_)

from data.Teeth_dataset import Teeth_Dataset
from data.CelebAF_dataset import CelebAF_Dataset
from models.cvae import Conditional_VAE
from loss import LossComputer

dataset_dict = {'Teeth': Teeth_Dataset,
                'CelebAF': CelebAF_Dataset,}
model_dict = {'cvae': Conditional_VAE}

class Train(object):
    def __init__(self):
        self.opt = self._parse_args()
        self.modes = ['train', 'valid']
        self.reporter = None
        # self.device = 'cpu'
        self.device = torch.device('cuda:{}'.format(self.opt.gpu) if torch.cuda.is_available() else 'cpu')
        print('[*] Using device: {}'.format(self.device))

    def _parse_args(self):
        self.arg_parser = argparse.ArgumentParser()
        # network architecture
        self.arg_parser.add_argument('--Adims', type=int, nargs='+', default=[32, 64, 128, 128, 128])
        self.arg_parser.add_argument('--zdim', type=int, default=128)
        self.arg_parser.add_argument('--Sdims', type=int, nargs='+', default=[32, 64, 128, 128, 128])#[8, 16, 32, 32, 32])
        self.arg_parser.add_argument('--cdim', type=int, default=128)
        self.arg_parser.add_argument('--Gdims', type=int, nargs='+', default=[32, 64, 128, 128, 128])
        self.arg_parser.add_argument('--Eksize', type=int, default=5)
        self.arg_parser.add_argument('--Epadding', type=int, default=2)
        self.arg_parser.add_argument('--Gksize', type=int, default=4)
        self.arg_parser.add_argument('--Gpadding', type=int, default=1)
        self.arg_parser.add_argument('--Gstride', type=int, default=2)

        # runtime settings
        self.arg_parser.add_argument('--run-name', type=str, default=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        self.arg_parser.add_argument('--model-fn', type=str, default='cvae')
        self.arg_parser.add_argument('--gpu', type=int, default=0)
        self.arg_parser.add_argument('--seed', type=int, default=None)
        self.arg_parser.add_argument('--no-bypass', type=bool, default=True)
        self.arg_parser.add_argument('--print-net', type=bool, default=False)
        self.arg_parser.add_argument('--batch-size', type=int, default=32)
        self.arg_parser.add_argument('--weight-decay', type=float, default=-1)
        self.arg_parser.add_argument('--lr', type=float, default=1e-3)
        self.arg_parser.add_argument('--lr-step', type=int, default=50)
        self.arg_parser.add_argument('--gamma', type=float, default=0.8)
        self.arg_parser.add_argument('--epochs', type=int, default=200)
        self.arg_parser.add_argument('--save-freq', type=int, default=10)
        self.arg_parser.add_argument('--report-freq', type=int, default=2)
        self.arg_parser.add_argument('--note', type=str, default='')

        # loss hyperparameters
        self.arg_parser.add_argument('--perc-net', type=str, default='vgg19')
        self.arg_parser.add_argument('--perc-idx', type=int, nargs='+', default=[3, 8, 17])
        self.arg_parser.add_argument('--perc-weight', type=float, default=1e-2)
        self.arg_parser.add_argument('--kl-weight', type=float, default=0.5)
        self.arg_parser.add_argument('--l1-weight', type=float, default=1.0)

        # data preprocessing
        self.arg_parser.add_argument('--dataset-fn', type=str, default='CelebAF')
        self.arg_parser.add_argument('--dataset-root', type=str, default='/tmp/CelebA/')
        self.arg_parser.add_argument('--image-size', type=int, default=256)

        ## this part is quite flexible
        self.add_arg()
        opt, _ = self.arg_parser.parse_known_args()
        return opt

    def add_arg(self):
        return

    def checkpoint_prefix(self):
        return 'model'

    def preprocess(self):
        if self.opt.run_name is None or self.opt.run_name == "":
            raise Exception('No run_name specified!')
        if self.opt.model_fn is None or self.opt.model_fn == "":
            raise Exception('No model_fn specified!')
        if self.opt.dataset_fn is None or self.opt.dataset_fn == "":
            raise Exception('No dataset_fn specified!')
        if self.opt.dataset_root is None or self.opt.dataset_root == "":
            raise Exception('No dataset_root specified!')

        self.opt.log_dir = 'logs/%s' % self.opt.run_name
        self.continue_run  = os.path.exists(os.path.join(self.opt.log_dir, 'options_train.pkl'))
        self.continue_fold = os.path.exists(os.path.join(self.opt.log_dir, 'models', '%s-latest.pth'%self.ckpt_prefix))

        # random.seed(self.opt.seed)
        # np.random.seed(self.opt.seed)
        # torch.manual_seed(self.opt.seed)

    def forward_batch(self, batch_data, is_train):
        ori_images, silhouettes = batch_data
        ori_images = ori_images.to(self.device)
        silhouettes = silhouettes.to(self.device)

        edges, masks     = torch.split(silhouettes, [2, 1], dim=1)
        ori_images_mask  = ori_images * masks
        ori_images_rmask = ori_images * (1-masks)

        syn_images, _, means, log_stds, loss_dict = self.net(ori_images_mask, ori_images_rmask, silhouettes, masks, ori_images)
        return syn_images

    def run(self):
        self.ckpt_prefix = self.checkpoint_prefix()
        self.preprocess()

        self.datasets = {m: dataset_dict[self.opt.dataset_fn](m, self.opt) for m in self.modes}

        self.dataloaders = {m: DataLoader(
            dataset=self.datasets[m],
            batch_size=self.opt.batch_size,
            num_workers=2,
            drop_last=True if m == 'train' else False,
            shuffle=True
        ) for m in self.modes}

        self.net = model_dict[self.opt.model_fn](self.device, self.opt)
        # load checkpoint data if possible
        load_dir = os.path.join(_project_folder_ ,'%s/models/' % self.opt.log_dir)
        print('[*] Load status data from %s' % load_dir)
        data = self.net.load(load_dir=load_dir,
                                prefix=self.ckpt_prefix,
                                mode='best',)
                            #  optimizer=self.optimizer,
                            #  lr_exp_scheduler=self.lr_exp_scheduler)
        
        cnt = 0
        for batch_data in self.dataloaders['valid']:
            ori_images, silhouettes = batch_data
            syn_images = self.forward_batch(batch_data, False)

            for ori, syn in zip(ori_images, syn_images):
                ori=cv2.cvtColor(ori.cpu().numpy().transpose(1,2,0)*256, cv2.COLOR_RGB2BGR)
                cv2.imwrite('/tmp/test/ori{}.jpg'.format(cnt), ori)
                syn=cv2.cvtColor(syn.cpu().detach().numpy().transpose(1,2,0)*256, cv2.COLOR_RGB2BGR)
                cv2.imwrite('/tmp/test/syn{}.jpg'.format(cnt), syn)

                cnt += 1
            break

if __name__ == '__main__':
    trainer = Train()
    trainer.run()
