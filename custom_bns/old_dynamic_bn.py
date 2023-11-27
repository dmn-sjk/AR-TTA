"""
Based on MECTA code: https://github.com/SonyResearch/MECTA
and https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py.
"""


import torch
from torch.nn import BatchNorm2d

import torch
from torch.nn.modules.batchnorm import BatchNorm2d
from utils.utils import gauss_symm_kl_divergence


class DynamicBN(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, beta=0.1,
                 init_beta=None,
                 bn_dist_scale=1., 
                 name='dynBN', 
                 smoothing_beta=0.2,
                 ):
        super(DynamicBN, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self.name = name
        self.beta = beta
        self.init_beta = self.beta if init_beta is None else init_beta

        self.bn_dist_scale = bn_dist_scale
        self.dist_metric = gauss_symm_kl_divergence

        self.smoothing_beta = smoothing_beta
        
        self.saved_running_mean = None
        self.saved_running_var = None
        
        self.test_mean = None
        self.test_var = None

        self.dynamic_bn_on = False

    def save_running_stats(self):
        self.saved_running_mean = self.running_mean.clone().cuda()
        self.saved_running_var = self.running_var.clone().cuda()
    
    def turn_dynamic_bn_on(self):
        self.save_running_stats()
        self.dynamic_bn_on = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum


        batch_var, batch_mean = torch.var_mean(input, dim=(0,2,3), unbiased=False)
        
        self.test_mean = batch_mean
        self.test_var = batch_var
        
        # calculate running estimates
        if self.training:
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * batch_mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * batch_var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
            mean = batch_mean
            var = batch_var

        elif self.dynamic_bn_on:
            self.running_mean = self.running_mean.detach()
            self.running_var = self.running_var.detach()

            dist = self.dist_metric(batch_mean, batch_var,
                                        self.running_mean, self.running_var, 
                                        eps=1e-3)  # self.eps) Small eps can reduce the sensitivity to unstable small variance.
            # dist = self.dist_metric(batch_mean, batch_var,
            #                             self.saved_running_mean.to(batch_mean.device), 
            #                             self.saved_running_var.to(batch_mean.device), 
            #                             eps=1e-3)  # self.eps) Small eps can reduce the sensitivity to unstable small variance.
            new_beta = 1. - torch.exp(- self.bn_dist_scale * dist.mean())
            # beta = 1. - torch.exp(- self.bn_dist_scale * dist.mean())

            # ema beta
            beta = (1 - self.smoothing_beta) * self.beta + self.smoothing_beta * new_beta

            # update beta
            self.beta = beta.item() # if hasattr(beta, 'item') else beta

            if beta < 1.:  # accumulate
                if self.saved_running_var is not None and self.saved_running_mean is not None:
                    self.running_mean.data.copy_(self.saved_running_mean).mul_((1-beta)).add_(batch_mean.mul(beta))
                    self.running_var.data.copy_(self.saved_running_var).mul_((1-beta)).add_(batch_var.mul(beta))
                else:
                    self.running_mean.data.copy_(self.running_mean).mul_((1-beta)).add_(batch_mean.mul(beta))
                    self.running_var.data.copy_(self.running_var).mul_((1-beta)).add_(batch_var.mul(beta))
            else:
                self.running_mean.data.copy_(batch_mean)
                self.running_var.data.copy_(batch_var)
                
            mean = self.running_mean
            var = self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input
