"""
Based on MECTA code: https://github.com/SonyResearch/MECTA
and https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py.
"""


from typing import Any, Mapping
import torch
from torch.nn import BatchNorm2d

import torch
from torch.nn.modules.batchnorm import BatchNorm2d
from utils.utils import gauss_symm_kl_divergence


class MectaBN(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, beta=0.1,
                 init_beta=None,
                 bn_dist_scale=1., 
                 name='dynBN', 
                 smoothing_beta=0.2,
                 ):
        super(MectaBN, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self.name = name

        self.beta = beta
        self.init_beta = self.beta if init_beta is None else init_beta

        self.bn_dist_scale = bn_dist_scale
        self.dist_metric = gauss_symm_kl_divergence

        self.test_mean = None
        self.test_var = None
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        assert self.training == False

        if self.training:
            return super(MectaBN, self).forward(input)
        
        if input.shape[0] == 1 and input.shape[2] == 1 and input.shape[3] == 1:
            raise ValueError(f"Cannot use batch norm if input shape: {input.shape}")
        
        batch_var, batch_mean = torch.var_mean(input, dim=(0,2,3), unbiased=False)
        
        # only for visualization
        self.test_mean = batch_mean
        self.test_var = batch_var

        dist = self.dist_metric(batch_mean, batch_var,
                                    self.running_mean, self.running_var, 
                                    eps=1e-3)  # self.eps) Small eps can reduce the sensitivity to unstable small variance.
        beta = 1. - torch.exp(- self.bn_dist_scale * dist.mean())

        # update beta
        self.beta = beta.item() # if hasattr(beta, 'item') else beta

        if beta < 1.:  # accumulate
            self.running_mean.data.copy_(self.running_mean).mul_((1-beta)).add_(batch_mean.mul(beta))
            self.running_var.data.copy_(self.running_var).mul_((1-beta)).add_(batch_var.mul(beta))
        else:
            self.running_mean.data.copy_(batch_mean)
            self.running_var.data.copy_(batch_var)
        
        return super(MectaBN, self).forward(input)
