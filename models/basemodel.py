from __future__ import division
from __future__ import print_function
import os
import torch
import glob

class BaseModel(object):

    def __init__(self):
        self._nets = list()
        self._net_names = list()
        self._train_flags = list()

    def __call__(self, *args):
        pass

    def register_nets(self, nets, names, train_flags):
        self._nets.extend(nets)
        self._net_names.extend(names)
        self._train_flags.extend(train_flags)

    def params(self, trainable, named=False, add_prefix=False):
        """
        :param trainable: True - params to train; False - all params
        :param named:
        :param add_prefix:
        :return:
        """

        def _get_net_params(_net, _net_name):
            if named:
                if add_prefix:
                    return [
                        (_net_name + '.' + _param_name, _param_data) for _param_name, _param_data in _net.named_parameters()
                    ]
                else:
                    return list(_net.named_parameters())
            else:
                return list(_net.parameters())

        res = list()
        for idx, net in enumerate(self._nets):
            net_flag = self._train_flags[idx]
            net_name = self._net_names[idx]

            if trainable:
                if net_flag:
                    res.extend(_get_net_params(net, net_name))
            else:
                res.extend(_get_net_params(net, net_name))
        return res

    def params_to_optimize(self, l2_weight_decay=1e-4, excludes=('bias',)):
        if l2_weight_decay > 0:
            # Use weight decay
            # https://blog.csdn.net/LoseInVain/article/details/81708474
            if excludes is None:
                excludes = list()

            decay_params = list()
            nondecay_params = list()

            named_params = self.params(True, named=True, add_prefix=False)
            for param_name, param_data in named_params:
                # search param_name in excludes using string op 'in'
                # all parameter data with their names including 'bias' will be put into decay_params
                use_decay = True
                for kw in excludes:
                    if kw in param_name:
                        use_decay = False
                        break
                if use_decay:
                    decay_params.append(param_data)
                else:
                    nondecay_params.append(param_data)
            return [{'params': decay_params, 'weight_decay': l2_weight_decay}, {'params': nondecay_params, 'weight_decay': 0}]
        else:
            # No weight decay
            return self.params(True, named=False, add_prefix=False)

    def print_params(self):
        print('[*] Model Parameters:')
        for nid, net in enumerate(self._nets):
            if self._train_flags[nid]:
                print('[*]  Trainable Module {}'.format(self._net_names[nid]))
            else:
                print('[*]  None-Trainable Module {}'.format(self._net_names[nid]))
            for name, param in net.named_parameters():
                print('[*]    {}: {}'.format(name, param.size()))
        print('[*] Model Size: {:.5f}M'.format(self.num_params() / 1e6))

    def num_params(self):
        return sum(p.numel() for p in self.params(False))

    def subnet_dict(self):
        return {self._net_names[i]: self._nets[i] for i in range(len(self._nets))}

    def save(self, save_dir, mode, prefix, best_epoch, best_loss, 
             curr_epoch=None, records=None, optimizer=None, lr_exp_scheduler=None):
        assert(mode == 'ckpt' or mode == 'best')
        state = {}
        # save nets and the best record
        for net, name in zip(self._nets, self._net_names):
            state[name] = net.state_dict()
            
        if mode == 'ckpt':
            torch.save(state, os.path.join(save_dir, '%s-epoch%d.pth' % (prefix, curr_epoch)))
            state['best_epoch']         = best_epoch
            state['best_loss']          = best_loss
            state['curr_epoch']         = curr_epoch
            state['records']            = records
            state['optimizer']          = optimizer.state_dict()
            state['lr_exp_scheduler']   = None if lr_exp_scheduler is None else lr_exp_scheduler.state_dict()
            torch.save(state, os.path.join(save_dir, '%s-latest.pth' % prefix))
        else:
            state['best_epoch']         = best_epoch
            state['best_loss']          = best_loss
            torch.save(state, os.path.join(save_dir, '%s-%s.pth' % (prefix, mode)))

    def load(self, load_dir, prefix, mode, optimizer=None, lr_exp_scheduler=None, epoch=None):
        assert(mode == 'latest' or mode == 'ckpt' or mode == 'best')
        if mode == 'ckpt':
            load_path = os.path.join(load_dir, '%s-epoch%d.pth' % (prefix, epoch))
        else:
            load_path = os.path.join(load_dir, '%s-%s.pth' % (prefix, mode))
        state = torch.load(load_path)
        for net, name in zip(self._nets, self._net_names):  
            net.load_state_dict(state[name])

        if mode == 'ckpt': return
        
        best_epoch = state['best_epoch']
        best_loss  = state['best_loss']
        if mode == 'best': return best_epoch, best_loss

        # mode == 'latest'
        init_epoch      = state['curr_epoch'] + 1
        records         = state['records']
        optimizer.load_state_dict(state['optimizer'])
        if lr_exp_scheduler is not None:
            lr_exp_scheduler.load_state_dict(state['lr_exp_scheduler'])
        return best_epoch, best_loss, init_epoch, records, optimizer, lr_exp_scheduler

    def train_mode(self):
        for net, train_flag in zip(self._nets, self._train_flags):
            if train_flag:
                # .train() and .eval() are different only when Dropout or BatchNorm are 
                # included in the network.
                net.train()
            else:
                net.eval()

    def eval_mode(self):
        for net in self._nets:
            net.eval()

    def to(self, device):
        for net in self._nets:
            net.to(device)
