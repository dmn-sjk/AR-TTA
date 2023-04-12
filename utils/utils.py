from torch import nn
import torch
import random
import numpy as np


def norm_params_unchanged(strategy, prev_norm_params):
        for m in strategy.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                if prev_norm_params['running_mean'] is not None:
                    if not torch.all(torch.eq(prev_norm_params['running_mean'], m.running_mean.cpu())):
                        print('running_mean NOT THE SAME')

                if prev_norm_params['running_var'] is not None:
                    if not torch.all(torch.eq(prev_norm_params['running_var'], m.running_var.cpu())):
                        print('running_var NOT THE SAME')

                prev_norm_params['running_mean'] = m.running_mean.cpu()
                prev_norm_params['running_var'] = m.running_var.cpu()

                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        if prev_norm_params['named'] is not None:
                            if not torch.all(torch.eq(prev_norm_params['named'], p.data.cpu())):
                                print('named NOT THE SAME')

                        prev_norm_params['named'] = p.data.cpu()
                        break
                break

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True