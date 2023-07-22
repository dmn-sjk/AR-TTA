"""
Based on MECTA code: https://github.com/SonyResearch/MECTA
and https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py.
"""


import torch
from torch import nn
from torch.nn import BatchNorm2d

from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states
import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn.functional as F


class DynamicBN(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, beta=0.1,
                 init_beta=None,
                 verbose=False, 
                 bn_dist_scale=1., 
                 name='bn', 
                 smoothing_beta=0.2,
                 ):
        super(DynamicBN, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self.name = name
        self.beta = beta
        self.init_beta = self.beta if init_beta is None else init_beta

        self.verbose = verbose

        self.bn_dist_scale = bn_dist_scale
        self.dist_metric = gauss_symm_kl_divergence

        self.smoothing_beta = smoothing_beta
        
        self.saved_running_mean = None
        self.saved_running_var = None

        self.dynamic_bn_on = False

    def save_running_stats(self):
        self.saved_running_mean = deepcopy(self.running_mean)
        self.saved_running_var = deepcopy(self.running_var)
    
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
            dist = self.dist_metric(batch_mean, batch_var,
                                        self.running_mean, self.running_var, 
                                        eps=1e-3)  # self.eps) Small eps can reduce the sensitivity to unstable small variance.
            new_beta = 1. - torch.exp(- self.bn_dist_scale * dist.mean())

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

# @torch.jit.script
def gauss_symm_kl_divergence(mean1, var1, mean2, var2, eps):
    # >>> out-place ops
    dif_mean = (mean1 - mean2) ** 2
    d1 = var1 + eps + dif_mean
    d1.div_(var2 + eps)
    d2 = (var2 + eps + dif_mean)
    d2.div_(var1 + eps)
    d1.add_(d2)
    d1.div_(2.).sub_(1.)
    # d1 = (var1 + eps + dif_mean) / (var2 + eps) + (var2 + eps + dif_mean) / (var1 + eps)
    return d1

def replace_bn(model, name=None, n_repalced=0, **abn_kwargs):
    copy_keys = ['eps', 'momentum', 'affine', 'track_running_stats']

    for mod_name, target_mod in model.named_children():
        # print(f"## inspect module: {name}.{mod_name}")
        if isinstance(target_mod, nn.BatchNorm2d) or isinstance(target_mod, nn.SyncBatchNorm):
            # print(target_mod)
            # print(f" Insert dynamicBN to ", name + '.' + mod_name)
            print(" Insert dynamicBN to ", mod_name)
            n_repalced += 1

            new_mod = DynamicBN(
                target_mod.num_features,
                **{k: getattr(target_mod, k) for k in copy_keys},
                **abn_kwargs,
                # name=f'{name}.{mod_name}'
            )
            new_mod.load_state_dict(target_mod.state_dict())
            setattr(model, mod_name, new_mod)
            new_mod.turn_dynamic_bn_on()
        else:
            n_repalced = replace_bn(
                target_mod, n_repalced=n_repalced, **abn_kwargs)
    return n_repalced

def count_bn(model: nn.Module):
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            cnt += 1
    return cnt


if __name__ == "__main__":
    def compare_bn(bn1, bn2):
        err = False
        if not torch.allclose(bn1.running_mean, bn2.running_mean):
            print('Diff in running_mean: {} vs {}'.format(
                bn1.running_mean, bn2.running_mean))
            err = True

        if not torch.allclose(bn1.running_var, bn2.running_var):
            print('Diff in running_var: {} vs {}'.format(
                bn1.running_var, bn2.running_var))
            err = True

        if bn1.affine and bn2.affine:
            if not torch.allclose(bn1.weight, bn2.weight):
                print('Diff in weight: {} vs {}'.format(
                    bn1.weight, bn2.weight))
                err = True

            if not torch.allclose(bn1.bias, bn2.bias):
                print('Diff in bias: {} vs {}'.format(
                    bn1.bias, bn2.bias))
                err = True

        if not err:
            print('All parameters are equal!')

    my_bn = DynamicBN(3, affine=True)
    bn = nn.BatchNorm2d(3, affine=True)

    compare_bn(my_bn, bn)  # weight and bias should be different
    # Load weight and bias
    my_bn.load_state_dict(bn.state_dict())
    compare_bn(my_bn, bn)

    # Run train
    for _ in range(10):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100) * scale + bias
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        # torch.allclose(out1, out2)
        # print(f"All close: {torch.allclose(out1, out2)}")
        print('Max diff: ', (out1 - out2).abs().max())

    # Run eval
    my_bn.eval()
    bn.eval()
    my_bn.turn_dynamic_bn_on()
    for _ in range(10):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100) * scale + bias
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        # print(f"All close: {torch.allclose(out1, out2)}")
        print('Max diff: ', (out1 - out2).abs().max())