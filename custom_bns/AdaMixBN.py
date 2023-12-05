"""
https://github.com/koncle/DomainAdaptor
"""


import torch.nn as nn
import torch
from typing import Any, Mapping


class AdaMixBN(nn.BatchNorm2d):
    # AdaMixBn cannot be applied in an online manner.
    def __init__(self, in_ch, lambd=None, transform=True, mix=True, idx=0, **kwargs):
        super(AdaMixBN, self).__init__(in_ch)
        self.lambd = lambd
        self.rectified_params = None
        self.transform = transform
        self.layer_idx = idx
        self.mix = mix
        
    def save_params(self):
        self.saved_weight = self.weight.data.clone().cuda()
        self.saved_bias = self.bias.data.clone().cuda()
        
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        incomp_keys = super().load_state_dict(state_dict, strict)
        self.save_params()
        return incomp_keys

    def get_retified_gamma_beta(self, lambd, src_mu, src_var, cur_mu, cur_var):
        C = src_mu.shape[1]
        new_gamma = (cur_var + self.eps).sqrt() / (lambd * src_var + (1 - lambd) * cur_var + self.eps).sqrt() * self.weight.view(1, C, 1, 1)
        new_beta = lambd * (cur_mu - src_mu) / (cur_var + self.eps).sqrt() * new_gamma + self.bias.view(1, C, 1, 1)
        return new_gamma.view(-1), new_beta.view(-1)

    def get_lambd(self, x, src_mu, src_var, cur_mu, cur_var):
        instance_mu = x.mean((2, 3), keepdims=True)
        instance_std = x.std((2, 3), keepdims=True)

        it_dist = ((instance_mu - cur_mu) ** 2).mean(1, keepdims=True) + ((instance_std - cur_var.sqrt()) ** 2).mean(1, keepdims=True)
        is_dist = ((instance_mu - src_mu) ** 2).mean(1, keepdims=True) + ((instance_std - src_var.sqrt()) ** 2).mean(1, keepdims=True)
        st_dist = ((cur_mu - src_mu) ** 2).mean(1)[None] + ((cur_var.sqrt() - src_var.sqrt()) ** 2).mean(1)[None]

        src_lambd = 1 - (st_dist) / (st_dist + is_dist + it_dist)

        src_lambd = torch.clip(src_lambd, min=0, max=1)
        return src_lambd

    def get_mu_var(self, x):
        C = x.shape[1]
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        cur_mu = x.mean((0, 2, 3), keepdims=True)
        cur_var = x.var((0, 2, 3), keepdims=True)

        lambd = self.get_lambd(x, src_mu, src_var, cur_mu, cur_var).mean(0, keepdims=True)

        if self.lambd is not None:
            lambd = self.lambd

        if self.transform:
            if self.rectified_params is None:
                new_gamma, new_beta = self.get_retified_gamma_beta(lambd, src_mu, src_var, cur_mu, cur_var)
                # self.test(x, lambd, src_mu, src_var, cur_mu, cur_var, new_gamma, new_beta)
                self.weight.data = new_gamma.data
                self.bias.data = new_beta.data
                self.rectified_params = new_gamma, new_beta
            return cur_mu, cur_var
        else:
            new_mu = lambd * src_mu + (1 - lambd) * cur_mu
            new_var = lambd * src_var + (1 - lambd) * cur_var
            return new_mu, new_var

    def forward(self, x):
        n, C, H, W = x.shape
        new_mu = x.mean((0, 2, 3), keepdims=True)
        new_var = x.var((0, 2, 3), keepdims=True)

        if self.training:
            return super(AdaMixBN, self).forward(x)
        
        if self.mix:
            self.weight.data = self.saved_weight.data.clone()
            self.bias.data = self.saved_bias.data.clone()
            self.reset()
            
            new_mu, new_var = self.get_mu_var(x)

            # Normalization with new statistics
            inv_std = 1 / (new_var + self.eps).sqrt()
            new_x = (x - new_mu) * (inv_std * self.weight.view(1, C, 1, 1)) + self.bias.view(1, C, 1, 1)
            
            return new_x

    def reset(self):
        self.rectified_params = None

    def test_equivalence(self, x):
        C = x.shape[1]
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        cur_mu = x.mean((0, 2, 3), keepdims=True)
        cur_var = x.var((0, 2, 3), keepdims=True)
        lambd = 0.9

        new_gamma, new_beta = self.get_retified_gamma_beta(x, lambd, src_mu, src_var, cur_mu, cur_var)
        inv_std = 1 / (cur_var + self.eps).sqrt()
        x_1 = (x - cur_mu) * (inv_std * new_gamma.view(1, C, 1, 1)) + new_beta.view(1, C, 1, 1)

        new_mu = lambd * src_mu + (1 - lambd) * cur_mu
        new_var = lambd * src_var + (1 - lambd) * cur_var
        inv_std = 1 / (new_var + self.eps).sqrt()
        x_2 = (x - new_mu) * (inv_std * self.weight.view(1, C, 1, 1)) + self.bias.view(1, C, 1, 1)
        assert (x_2 - x_1).abs().mean() < 1e-5
        return x_1, x_2