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
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from models.cvae import Conditional_VAE

# class Layout_Dataset(Dataset):
#     def __init__(self, imgs, masks):
#         super().__init__()
#         self.data = zip(imgs, masks)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.da

class Layout(object):
    def __init__(self):
        self.opt = self._parse_args()
        self.modes = ['train', 'valid']
        self.reporter = None
        self.device = torch.device('cuda:{}'.format(self.opt.gpu) if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        print('[*] Using device: {}'.format(self.device))

    def _parse_args(self):
        self.arg_parser = argparse.ArgumentParser()
        # network architecture
        self.arg_parser.add_argument('--Adims', type=int, nargs='+', default=[32, 64, 128, 128, 128])
        self.arg_parser.add_argument('--zdim', type=int, default=128)
        self.arg_parser.add_argument('--Sdims', type=int, nargs='+', default=[32, 64, 128, 128, 128])
        self.arg_parser.add_argument('--cdim', type=int, default=128)
        self.arg_parser.add_argument('--Gdims', type=int, nargs='+', default=[32, 64, 128, 128, 128])

        self.arg_parser.add_argument('--Ddims', type=int, nargs='+', default=[32, 64, 128, 128, 128, 128])
        self.arg_parser.add_argument('--ddim', type=int, nargs='+', default=1024)

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
        self.arg_parser.add_argument('--no-bypass', type=bool, default=False)
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
        self.arg_parser.add_argument('--kl-weight', type=float, default=1.0)
        self.arg_parser.add_argument('--l1-weight', type=float, default=1e-3)
        self.arg_parser.add_argument('--dis-weight', type=float, default=1.0)
        self.arg_parser.add_argument('--gen-weight', type=float, default=1.0)

        # data preprocessing
        self.arg_parser.add_argument('--dataset-fn', type=str, default='CelebAF')
        self.arg_parser.add_argument('--dataset-root', type=str, default='/tmp/CelebA/')
        self.arg_parser.add_argument('--image-size', type=int, default=256)

        ## this part is quite flexible
        opt, _ = self.arg_parser.parse_known_args()
        return opt
    
    def preproc(self):
        impath = os.path.join(self.opt.dataset_root, 'valid')
        dirs = os.listdir(impath)

        x_imgs = []
        y_masks = []
        y_masks_modify = []

        for d in dirs[:30]:
            img = cv2.imread(os.path.join(impath, d, 'Img.jpg'))
            mask = cv2.imread(os.path.join(impath, d, 'mask.jpg'))
            p2d = np.load(os.path.join(impath, d, 'bbox-ldmk.npz'))['ldmk']

            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            y = np.zeros_like(x)
            y_m = np.zeros_like(x)

            for pt in p2d.astype(np.int):
                cv2.circle(y, tuple(pt), 2, (255,255,255), -1)

            for i, pt in enumerate(p2d.astype(np.int)):
                if i+1 >=37 and i+1 <=48:
                    pt[1] += 15
                cv2.circle(y_m, tuple(pt), 2, (255,255,255), -1)
                
            y[:, :, 2] = mask[:, :, 2]
            y_m[:, :, 2] = mask[:, :, 2]

            
            # x = np.array(x).transpose((2,0,1))
            # y = np.array(y).transpose((2,0,1))
            # y_m = np.array(y_m).transpose((2,0,1))
        
            x = TF.to_tensor(TF.to_pil_image(x))
            y = TF.to_tensor(TF.to_pil_image(y))
            y_m = TF.to_tensor(TF.to_pil_image(y_m))
            
            y = (y > 0.6).float()
            y_m = (y_m > 0.6).float()
            
            print(x.shape)
            x_imgs.append(x)
            y_masks.append(y)
            y_masks_modify.append(y_m)

        ori_imgs = list(zip(x_imgs, y_masks))
        mdy_imgs = list(zip(x_imgs, y_masks_modify))
        return ori_imgs, mdy_imgs

    def forward_batch(self, batch_data):
        ori_images, silhouettes = batch_data
        ori_images = ori_images.to(self.device)
        silhouettes = silhouettes.to(self.device)

        edges, masks     = torch.split(silhouettes, [2, 1], dim=1)
        ori_images_mask  = ori_images * masks
        ori_images_rmask = ori_images * (1-masks)

        syn_images_f, syn_images_p, means, log_stds, loss_dict = self.net.calc_gen_loss(ori_images_mask, ori_images_rmask, silhouettes, masks)

        return syn_images_p + ori_images_rmask
    
    def wrt_images(self, path, imgs):
        imgs=cv2.cvtColor(imgs.cpu().numpy().transpose(1,2,0)*256, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, imgs)

    def run(self):
        dataset, dataset_m = self.preproc()
        dataset = DataLoader(dataset)
        dataset_m = DataLoader(dataset_m)

        self.net = Conditional_VAE(self.device, self.opt)
        self.opt.log_dir = os.path.join(os.path.dirname(__file__),'logs', self.opt.run_name)
        self.net.load(
            load_dir='%s/models' % self.opt.log_dir,
            prefix='model',
            mode='latest',
        )

        path = '/tmp/layout'
        cnt = 0
        for batch_data in dataset:
            ori_images, _= batch_data
            syn_images = self.forward_batch(batch_data)

            for ori, syn in zip(ori_images, syn_images):
                self.wrt_images(os.path.join(path, 'ori{}.jpg'.format(cnt)), ori)
                self.wrt_images(os.path.join(path, 'syn{}.jpg'.format(cnt)), syn.detach())
                cnt += 1

        cnt = 0
        for batch_data in dataset_m:
            syn_images = self.forward_batch(batch_data)

            for syn in syn_images:
                self.wrt_images(os.path.join(path, 'lsyn{}.jpg'.format(cnt)), syn.detach())
                cnt += 1

if __name__ == "__main__":
    item = Layout()
    item.run()
