from torch import nn
import torch
import random
import numpy as np
import os
import subprocess


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

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_experiment_name(cfg: dict) -> str:
    return f"{cfg['dataset']}_{cfg['benchmark']}_{cfg['method']}_{cfg['model']}_{cfg['run_name']}"

def get_experiment_folder(cfg: dict) -> str:
    return os.path.join(cfg['log_dir'], get_experiment_name(cfg))

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
