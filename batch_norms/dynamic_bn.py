"""
Based on MECTA code: https://github.com/SonyResearch/MECTA
and https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py.
"""


from typing import Any, Mapping
import torch
from torch.nn.modules.batchnorm import BatchNorm2d
from utils.utils import gauss_symm_kl_divergence


class DynamicBN(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, beta=0.1,
                 bn_dist_scale=1., 
                 name='dynBN', 
                 smoothing_beta=0.2,
                 **kwargs
                 ):
        super(DynamicBN, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self.name = name

        self.beta = beta

        self.bn_dist_scale = bn_dist_scale
        self.dist_metric = gauss_symm_kl_divergence

        self.smoothing_beta = smoothing_beta
        
        self.saved_running_mean = None
        self.saved_running_var = None
        
        self.test_mean = None
        self.test_var = None
        
        self.adapt_bn_stats = True
        
    def save_running_stats(self):
        self.saved_running_mean = self.running_mean.clone().cuda()
        self.saved_running_var = self.running_var.clone().cuda()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        incomp_keys = super().load_state_dict(state_dict, strict)
        self.save_running_stats()
        return incomp_keys
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        assert self.training == False
        assert self.track_running_stats == False

        if self.training:
            return super(DynamicBN, self).forward(input)
        
        if self.adapt_bn_stats:
            if self.saved_running_mean is None or self.saved_running_var is None:
                raise RuntimeError(f"Using {self.__name__} requires setting up saved_running_mean and saved_running_var")
            
            if input.shape[0] == 1 and input.shape[2] == 1 and input.shape[3] == 1:
                raise ValueError(f"Cannot use batch norm if input shape: {input.shape}")
            
            batch_var, batch_mean = torch.var_mean(input, dim=(0,2,3), unbiased=False)
            
            # only for visualization
            self.test_mean = batch_mean
            self.test_var = batch_var

            dist = self.dist_metric(batch_mean, batch_var,
                                        self.running_mean, self.running_var, 
                                        eps=1e-3)  # self.eps) Small eps can reduce the sensitivity to unstable small variance.
            new_beta = 1. - torch.exp(- self.bn_dist_scale * dist.mean())

            # ema beta
            beta = (1 - self.smoothing_beta) * self.beta + self.smoothing_beta * new_beta

            # update beta
            self.beta = beta.item() # if hasattr(beta, 'item') else beta

            if beta < 1.:  # accumulate
                self.running_mean.data.copy_(self.saved_running_mean).mul_((1-beta)).add_(batch_mean.mul(beta))
                self.running_var.data.copy_(self.saved_running_var).mul_((1-beta)).add_(batch_var.mul(beta))
            else:
                self.running_mean.data.copy_(batch_mean)
                self.running_var.data.copy_(batch_var)
            
        return super(DynamicBN, self).forward(input)
 