import torch
import torch.nn as nn
from .basemodel import BaseModel

class A_Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dims = [3] + opt.Adims + [opt.zdim*2]
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=opt.Eksize, padding=opt.Epadding),
            nn.BatchNorm2d(self.dims[i+1]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        ) for i in range(len(self.dims)-1)])
        
    def forward(self, x):
        # feature_maps = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # if i < len(self.layers) - 1: feature_maps.append(x)

        zdim = self.dims[-1] // 2
        mean    = x[:, :zdim, :, :]
        log_std = x[:, zdim:, :, :]
        std = torch.exp(0.5 * log_std)

        def _get_KLD_loss(a_means, a_log_stds):
            loss = -0.5 * torch.sum(1 + a_log_stds - a_means.pow(2) - a_log_stds.exp())
            return loss

        kld_loss = _get_KLD_loss(mean, log_std)

        return mean, std, log_std, kld_loss

class S_Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dims = [3] + opt.Sdims + [opt.cdim]
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=opt.Eksize, padding=opt.Epadding),
            nn.BatchNorm2d(self.dims[i+1]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        ) for i in range(len(self.dims)-1)])

    def forward(self, y):
        feature_maps = []
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers) - 1: feature_maps.append(y)
        
        return y, feature_maps 

import torchvision.models as models

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inDims = [opt.zdim + opt.cdim] + [2 * i for i in opt.Gdims[::-1]]
        self.outDims = opt.Gdims[::-1] + [3]
        self.layers = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(self.inDims[i], self.outDims[i], kernel_size=opt.Gksize, padding=opt.Gpadding, stride=opt.Gstride),
            nn.BatchNorm2d(self.outDims[i]),
            nn.ReLU() if i < len(self.inDims)-1 else nn.Sigmoid(),
        ) for i in range(len(self.inDims))])

        self.perc_idx = opt.perc_idx
        if opt.perc_net == 'vgg19':
            self.net = models.vgg19(pretrained=True).features
        else:
            raise Exception('[E] such perception network not implemented!')
    
    def forward(self, z, c, feature_maps, x_m, m):
        if feature_maps is None:
            feature_maps = [None] * (len(self.dims)-2)

        z = torch.cat([z, c], axis=1)
        for feature, layer in zip(feature_maps[::-1], self.layers[:-1]):
            z = layer(z)
            # if feature is not None:
            assert(z.size() == feature.size())
            z = torch.cat([z, feature], axis=1)
        
        x = self.layers[-1](z)
        x = x * m

        def _get_L1_loss(ori_images, syn_images, masks):
            layer_weight = 1. / torch.max(torch.ones([masks.size(0)]).cuda(), torch.sum(masks, axis=[1,2,3]))
            loss = torch.sum(torch.sum(torch.abs(ori_images-syn_images), axis=[1, 2, 3]) * layer_weight)
            return loss

        def _get_Perc_loss(ori_images, syn_images, masks):
            o = ori_images
            s = syn_images

            loss = 0
            cnt = 0
            for i, layer in enumerate(self.net):
                o = layer(o)
                s = layer(s)

                if i in self.perc_idx:
                    layer_weight = 1. / torch.max(torch.ones([masks.size(0)]).cuda(), torch.sum(masks, axis=[1,2,3])) / 4**cnt
                    layer_loss = torch.sum(torch.sum(torch.abs(o-s), axis=[1, 2, 3]) * layer_weight)
                    loss += layer_loss

                    cnt += 1
                    if cnt == len(self.perc_idx): break
            return loss

        l1_loss = _get_L1_loss(x_m, x, m)
        perc_loss = _get_Perc_loss(x_m, x, m)

        return x, perc_loss, l1_loss

class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dims = [3] + opt.Ddims

        img = torch.rand((opt.batch_size, 3, opt.image_size, opt.image_size))

        self.layers = [nn.Sequential(
            nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=opt.Eksize, padding=opt.Epadding),
            nn.BatchNorm2d(self.dims[i+1]),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2)
        ) for i in range(len(self.dims)-1)]

        for layer in self.layers:
            img = layer(img)
        shape = list(img.shape)
        size = shape[1]*shape[2]*shape[3]

        self.layers = self.layers + [nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=size, out_features=opt.ddim),
            nn.LeakyReLU(),
        )]
        
        # img = self.layers[-1](img)
        # self.mf_shape = img.shape

        self.layers = self.layers + [nn.Sequential(
            nn.Linear(in_features=opt.ddim, out_features=1),
        )]

        # img = self.layers[-1](img)
        # self.label_shape = img.shape

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                mean_feature = x
            x = layer(x)

        return mean_feature, x

class Conditional_VAE(BaseModel):
    def __init__(self, device, opt):
        super().__init__()
        self.device = device
        self.no_bypass = opt.no_bypass

        self.a_encoder = A_Encoder(opt)
        self.s_encoder = S_Encoder(opt)
        self.generator = Generator(opt)
        self.discriminator = Discriminator(opt)
        # self.net = nn.DataParallel(self.net)
        self.a_encoder = nn.DataParallel(self.a_encoder)
        self.s_encoder = nn.DataParallel(self.s_encoder)
        self.generator = nn.DataParallel(self.generator)
        self.discriminator = nn.DataParallel(self.discriminator)


        self.batch_size = opt.batch_size
        self.perc_weight = opt.perc_weight
        self.kl_weight = opt.kl_weight
        self.l1_weight = opt.l1_weight

        self.dis_weight = opt.dis_weight
        self.gen_weight = opt.gen_weight

        nets=[self.a_encoder, self.s_encoder, self.generator, self.discriminator]
        names=['AE', 'SE', 'G', 'D']
        train_flags=[True, True, True, True]
        self.register_nets(nets, names, train_flags)
        self.to(self.device)

    def calc_gen_loss(self, x_m, x_mo, y, m):
        
        loss_dict= {}
        
        # Encode appearance and structure independently
        mean, std, log_std, kld_loss = self.a_encoder(x_m)
        c, feature_maps = self.s_encoder(y)

        # Randomly sample z~G(means, stds)
        eps = torch.randn(mean.size()).to(self.device)
        z = eps * std + mean   
        # print(z.shape, c.shape)

        # Generate images
        r, perc_loss, l1_loss = self.generator(z, c, feature_maps, x_m, m)
        f, _, _ = self.generator(eps, c, feature_maps, x_m, m)
    
        # Discriminator
        mf_r , _ = self.discriminator(x_m) 
        mf_f2 , _ = self.discriminator(f) 
        gen_loss = torch.sum((mf_r - mf_f2).pow(2))

        # loss_dict['dis_loss'] = self.dis_weight * dis_loss / self.batch_size
        loss_dict['gen_loss'] = self.gen_weight * gen_loss / self.batch_size
        # loss_dict['gen_loss'] = 0
        loss_dict['kld_loss'] = self.kl_weight * torch.sum(kld_loss) / self.batch_size
        loss_dict['perc_loss'] = self.perc_weight * torch.sum(perc_loss) / self.batch_size
        loss_dict['l1_loss'] = self.l1_weight * torch.sum(l1_loss) / self.batch_size

        loss_dict['content_loss'] = loss_dict['kld_loss'] + loss_dict['perc_loss'] + loss_dict['l1_loss'] + loss_dict['gen_loss']

        return r, f, mean, log_std, loss_dict
    
    def calc_dis_loss(self, x_m, f1, f2):

        _, label_r = self.discriminator(x_m)
        _, label_f1 = self.discriminator(f1) 
        _, label_f2 = self.discriminator(f2) 

        get_dis_loss = nn.BCEWithLogitsLoss()

        dis_loss = get_dis_loss(label_r, torch.ones_like(label_r))
        dis_loss += get_dis_loss(label_f1, torch.zeros_like(label_r)) * 0.5
        dis_loss += get_dis_loss(label_f2, torch.zeros_like(label_r)) * 0.5

        dis_loss = self.dis_weight * dis_loss
        return dis_loss

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--Adims', type=int, nargs='+', default=[32, 64, 128, 128, 128])
    arg_parser.add_argument('--zdim', type=int, default=128)
    arg_parser.add_argument('--Sdims', type=int, nargs='+', default=[8, 16, 32, 32, 32])
    arg_parser.add_argument('--cdim', type=int, default=32)
    arg_parser.add_argument('--Eksize', type=int, default=5)
    arg_parser.add_argument('--Epadding', type=int, default=2)
    arg_parser.add_argument('--Gksize', type=int, default=4)
    arg_parser.add_argument('--Gpadding', type=int, default=1)
    arg_parser.add_argument('--Gstride', type=int, default=2)

    arg_parser.add_argument('--gpu', type=int, default=0)
    arg_parser.add_argument('--no-bypass', type=bool, default=False)
    arg_parser.add_argument('--print-net', type=bool, default=False)
    opt = arg_parser.parse_args()

    device = torch.device('cuda:{}'.format(opt.gpu) if torch.cuda.is_available() else 'cpu')
    x = torch.zeros([10,3,256,256]).to(device)
    y = torch.zeros([10,3,256,256]).to(device)
    m = torch.ones( [10,1,256,256]).to(device)
    x_m = x * m
    x_mo = x * (1-m)
    cvae = Conditional_VAE(device, opt)
    if opt.print_net: cvae.print_params()
    r, r_m, mean, std = cvae(x_m, x_mo, y, m)
    print(r.shape)
