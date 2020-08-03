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

        self.opt.log_dir = 'logs/%s/' % self.opt.run_name
        self.continue_run  = os.path.exists(os.path.join(self.opt.log_dir, 'options_train.pkl'))
        self.continue_fold = os.path.exists(os.path.join(self.opt.log_dir, 'models', '%s-latest.pth'%self.ckpt_prefix))

        if not self.continue_run:
            os.makedirs(os.path.join(self.opt.log_dir, 'models'), 0o777, exist_ok=True)

            # Reset random seed
            if self.opt.seed is None:
                self.opt.seed = random.randint(0, 2**31 - 1)

            # Save arguments
            with open(os.path.join(self.opt.log_dir, 'options_train.pkl'), 'wb') as fh:
                pickle.dump(self.opt, fh)
            with open(os.path.join(self.opt.log_dir, 'options_train.json'), 'w') as fh:
                fh.write(json.dumps(vars(self.opt), sort_keys=True, indent=4))
        else:
            # is_continue is True
            with open(os.path.join(self.opt.log_dir, 'options_train.pkl'), 'rb') as fh:
                self.opt = pickle.load(fh)

        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)

    def forward_batch(self, batch_data, is_train):
        ori_images, silhouettes = batch_data
        ori_images = ori_images.to(self.device)
        silhouettes = silhouettes.to(self.device)

        edges, masks     = torch.split(silhouettes, [2, 1], dim=1)
        ori_images_mask  = ori_images * masks
        ori_images_rmask = ori_images * (1-masks)

        with torch.set_grad_enabled(is_train):
            if is_train:
                self.optimizer.zero_grad()
            syn_images, _, means, log_stds, loss_dict = self.net(ori_images_mask, ori_images_rmask, silhouettes, masks, ori_images)
            # loss_dict = self.loss_computer.get_loss(ori_images, syn_images, masks, means, log_stds)
            Content_loss = loss_dict['content_loss']
            if is_train:
                Content_loss.backward()
                self.optimizer.step()

        return loss_dict, syn_images

    def run(self):
        self.ckpt_prefix = self.checkpoint_prefix()
        self.preprocess()
        if self.opt.report_freq > 0:
            self.reporter = SummaryWriter(logdir='%s/summary' % self.opt.log_dir)

        self.datasets = {m: dataset_dict[self.opt.dataset_fn](m, self.opt) for m in self.modes}
        print('[*] Dataset [%s] ready' % self.opt.dataset_fn)

        self.dataloaders = {m: DataLoader(
            dataset=self.datasets[m],
            batch_size=self.opt.batch_size,
            num_workers=2,
            drop_last=True if m == 'train' else False,
            shuffle=True
        ) for m in self.modes}

        self.net = model_dict[self.opt.model_fn](self.device, self.opt)
        # if self.opt.print_net:
        self.net.print_params()
        print('[*] Model [%s] ready' % self.opt.model_fn)

        self.optimizer = optim.Adam(self.net.params_to_optimize(self.opt.weight_decay),
                                    lr=self.opt.lr,
                                    betas=(0.9, 0.999))

        if self.opt.lr_step > 0:
            self.lr_exp_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.opt.lr_step, gamma=self.opt.gamma)
        else:
            self.lr_exp_scheduler = None

        self.loss_computer = LossComputer(self.device, self.opt)

        # load checkpoint data if possible
        if self.continue_fold:
            print('[*] Load status data from %s/models/%s-latest.pth' % (self.opt.log_dir, self.ckpt_prefix))
            data = self.net.load(load_dir='%s/models' % self.opt.log_dir,
                                 prefix=self.ckpt_prefix,
                                 mode='latest',
                                 optimizer=self.optimizer,
                                 lr_exp_scheduler=self.lr_exp_scheduler)
            best_epoch, best_loss, init_epoch, records, self.optimizer, self.lr_exp_scheduler = data
        else:
            init_epoch     = 1
            best_loss      = math.inf
            best_epoch     = -1
            records        = {m: [] for m in self.modes}

        for epoch in range(init_epoch, self.opt.epochs+1):
            print('=' * 40)
            print('[*] Epoch {}/{}'.format(epoch, self.opt.epochs))

            for mode in self.modes:
                is_train = mode == 'train'
                print('[*] Start {} mode'.format(mode))
                batches = len(self.dataloaders[mode])

                if is_train:
                    self.net.train_mode()
                else:
                    self.net.eval_mode()

                Content_loss = 0
                KLD_loss = 0
                Perc_loss = 0
                L1_loss = 0

                pbar = tqdm.tqdm(total=batches)
                for batch_data in self.dataloaders[mode]:
                    loss_dict, syn_images = self.forward_batch(batch_data, is_train)

                    Content_loss += loss_dict['content_loss'].cpu().item()
                    KLD_loss += loss_dict['kld_loss'].cpu().item()
                    Perc_loss += loss_dict['perc_loss'].cpu().item()
                    L1_loss += loss_dict['l1_loss'].cpu().item()

                    pbar.update()
                pbar.close()

                KLD_loss        /= batches
                Perc_loss       /= batches
                L1_loss         /= batches
                Content_loss    /= batches

                print('[*] {} loss: {:.4f} (L1 Loss: {:.4f})'.format(mode, Content_loss, L1_loss))
                records[mode].append((Content_loss, L1_loss, Perc_loss, KLD_loss))

                if epoch % self.opt.report_freq == 0 or epoch == self.opt.epochs or epoch == 1:
                    # report loss to tensorboard
                    if self.reporter is not None:
                        self.reporter.add_scalar('%s_Content_loss'%mode, Content_loss, epoch)
                        self.reporter.add_scalar('%s_KLD_loss'%mode, KLD_loss, epoch)
                        self.reporter.add_scalar('%s_L1_loss'%mode, L1_loss, epoch)
                        self.reporter.add_scalar('%s_Perc_loss'%mode, Perc_loss, epoch)

                # update best record
                if mode == 'valid' and Content_loss < best_loss:
                    best_loss  = Content_loss
                    best_epoch = epoch

            # end of epoch
            if self.lr_exp_scheduler is not None:
                self.lr_exp_scheduler.step()
            
            # save ckpt model if needed
            if epoch % self.opt.save_freq == 0 or epoch == self.opt.epochs:
                print('[*] Save ckpt and latest model at epoch %d.' % epoch)
                self.net.save(save_dir='%s/models' % self.opt.log_dir,
                              mode='ckpt',
                              prefix=self.ckpt_prefix,
                              best_epoch=best_epoch,
                              best_loss=best_loss,
                              curr_epoch=epoch,
                              records=records,
                              optimizer= self.optimizer,
                              lr_exp_scheduler= self.lr_exp_scheduler)

            # save best model if needed
            if epoch == best_epoch:
                print('[*] Save best model at epoch %d.' % epoch)
                self.net.save(save_dir='%s/models' % self.opt.log_dir,
                              mode='best',
                              prefix=self.ckpt_prefix,
                              best_epoch=best_epoch,
                              best_loss=best_loss)

        records['best'] = (best_epoch, best_loss)
        print('[*] Best loss: {:.4f}, corresponding epoch: {}'.format(best_loss, best_epoch))
        record_path = os.path.join(self.opt.log_dir, 'records_train_%s.json'%self.ckpt_prefix)
        with open(record_path, 'w') as fh:
            fh.write(json.dumps(records, sort_keys=True, indent=4))
        
        if self.reporter is not None:
            self.reporter.close()
        return

if __name__ == '__main__':
    trainer = Train()
    trainer.run()
