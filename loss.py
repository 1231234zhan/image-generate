import torch
import torchvision.models as models
import numpy as np

class LossComputer():
    def __init__(self, device, opt):
        self.device = device
        if opt.perc_net == 'vgg19':
            self.net = models.vgg19(pretrained=True).features
        else:
            raise Exception('[E] such perception network not implemented!')
        self.net.to(self.device).eval()
        
        self.perc_idx = opt.perc_idx
        self.perc_weight = opt.perc_weight
        self.kl_weight = opt.kl_weight
        self.l1_weight = opt.l1_weight

    def get_loss(self, ori_images, syn_images, masks, a_means, a_log_stds):
        loss_dict = {}
        loss_dict['kld_loss']   = self._get_KLD_loss(a_means, a_log_stds)
        loss_dict['perc_loss']  = self._get_Perc_loss(ori_images, syn_images, masks)
        loss_dict['l1_loss']    = self._get_L1_loss(ori_images, syn_images, masks)

        loss_dict['content_loss'] = loss_dict['kld_loss'] + loss_dict['perc_loss'] + loss_dict['l1_loss']
        return loss_dict

    def _get_KLD_loss(self, a_means, a_log_stds):
        loss = -0.5 * torch.mean(1 + a_log_stds - a_means.pow(2) - a_log_stds.exp())
        return loss * self.kl_weight

    def _get_Perc_loss(self, ori_images, syn_images, masks):
        o = ori_images
        s = syn_images

        loss = 0
        cnt = 0
        for i, layer in enumerate(self.net):
            o = layer(o)
            s = layer(s)

            if i in self.perc_idx:
                layer_weight = 1. / torch.max(torch.ones([masks.size(0)]).to(self.device), torch.sum(masks, axis=[1,2,3])) / 4**cnt
                layer_loss = torch.mean(torch.sum(torch.abs(o-s), axis=[1, 2, 3]) * layer_weight)
                loss += layer_loss

                cnt += 1
                if cnt == len(self.perc_idx): break
        return loss * self.perc_weight

    def _get_L1_loss(self, ori_images, syn_images, masks):
        layer_weight = 1. / torch.max(torch.ones([masks.size(0)]).to(self.device), torch.sum(masks, axis=[1,2,3]))
        loss = torch.mean(torch.sum(torch.abs(ori_images-syn_images), axis=[1, 2, 3]) * layer_weight)
        return loss * self.l1_weight

if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--perc-net', type=str, default='vgg19')
    arg_parser.add_argument('--perc-idx', type=int, nargs='+', default=[3, 8, 17])
    arg_parser.add_argument('--perc-weight', type=float, nargs='+', default=1e-2)
    arg_parser.add_argument('--kl-weight', type=float, default=0.5)
    arg_parser.add_argument('--l1-weight', type=float, default=1.0)

    arg_parser.add_argument('--gpu', type=int, default=0)
    arg_parser.add_argument('--print-net', type=bool, default=False)
    opt = arg_parser.parse_args()

    device = torch.device('cuda:{}'.format(opt.gpu) if torch.cuda.is_available() else 'cpu')
    X = torch.randn([10,3,256,256]).to(device)
    R = torch.randn([10,3,256,256]).to(device)
    M = torch.zeros([10,1,256,256]).to(device)
    M[:, :, 100:200, 100:200] = 1
    mean   = torch.zeros([10,128,4,4]).to(device)
    logstd = torch.zeros([10,128,4,4]).to(device)

    loss_computer = LossComputer(device, opt)

    with torch.set_grad_enabled(False):
        loss_dict = loss_computer.get_loss(X, R, M, mean, logstd)  

    for key, val in loss_dict.items():
        print(key, val.cpu().item())