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
        feature_maps = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1: feature_maps.append(x)

        zdim = self.dims[-1] // 2
        mean    = x[:, :zdim, :, :]
        log_std = x[:, zdim:, :, :]
        std = torch.exp(0.5 * log_std)

        return mean, std, log_std, feature_maps

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
        for layer in self.layers:
            y = layer(y)
        return y 

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dims = [opt.zdim + opt.cdim] + opt.Adims[::-1] + [3]
        self.layers = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(self.dims[i], self.dims[i+1], kernel_size=opt.Gksize, padding=opt.Gpadding, stride=opt.Gstride),
            nn.BatchNorm2d(self.dims[i+1]),
            nn.ReLU() if i < len(self.dims)-2 else nn.Sigmoid(),
        ) for i in range(len(self.dims)-1)])
    
    def forward(self, z, c, feature_maps, x_mo, m):
        if feature_maps is None:
            feature_maps = [None] * (len(self.dims)-2)

        z = torch.cat([z, c], axis=1)
        for feature, layer in zip(feature_maps[::-1], self.layers[:-1]):
            z = layer(z)
            if feature is not None:
                assert(z.size() == feature.size())
                z += feature
        
        x = self.layers[-1](z)
        x = x * m + x_mo
        return x

class Conditional_VAE(BaseModel):
    def __init__(self, device, opt):
        super().__init__()
        self.device = device
        self.no_bypass = opt.no_bypass

        self.a_encoder = A_Encoder(opt)
        self.s_encoder = S_Encoder(opt)
        self.generator = Generator(opt)

        nets=[self.a_encoder, self.s_encoder, self.generator]
        names=['AE', 'SE', 'G']
        train_flags=[True, True, True]
        self.register_nets(nets, names, train_flags)
        self.to(self.device)

    def __call__(self, x_m, x_mo, y, m):
        # Encode appearance and structure independently
        mean, std, log_std, feature_maps = self.a_encoder(x_m)
        c = self.s_encoder(y)

        # Randomly sample z~G(means, stds)
        eps = torch.randn(mean.size()).to(self.device)
        z = eps * std + mean   
        # print(z.shape, c.shape)

        # Generate images
        r = self.generator(z, c, None if self.no_bypass else feature_maps, x_mo, m)
        r_m = r * m

        return r, r_m, mean, log_std
    
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
