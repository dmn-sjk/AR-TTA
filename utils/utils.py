from torch import nn
import torch
import random
import numpy as np
import os
import subprocess
import yaml
from copy import copy, deepcopy


# ref: https://github.com/Oldpan/Pytorch-Memory-Utils/blob/master/gpu_mem_track.py
dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}

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
    return f"{cfg['dataset']}_{cfg['method']}_{cfg['model']}_{cfg['run_name']}"

def get_experiment_folder(cfg: dict) -> str:
    return os.path.join(cfg['log_dir'], get_experiment_name(cfg))

def get_seed_folder(cfg: dict) -> str:
    return os.path.join(get_experiment_folder(cfg), 'seed' + str(cfg['curr_seed']))

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def split_up_model(model, model_name, dataset=None):
    if 'wideresnet' in model_name: 
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
    elif 'resnet' in model_name:
        if dataset == 'imagenetc':
            encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
            model = model.model
        else:
            encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        
    classifier = model.fc

    return encoder, classifier

def save_config(cfg, experiment_name):
    temp_cfg = copy(cfg)
    del temp_cfg['domains']
    
    os.makedirs(get_seed_folder(cfg), exist_ok = True)
    
    if cfg['curr_seed'] is not None:
        with open(os.path.join(get_seed_folder(cfg), 'domains.yaml'), 'w') as f:
            yaml.dump(cfg['domains'], f, default_flow_style=False)

        del temp_cfg['curr_seed']
        with open(os.path.join(get_experiment_folder(cfg), experiment_name + '_config.yaml'), 'w') as f:
            yaml.dump(temp_cfg, f, default_flow_style=False)
    else:
        with open(os.path.join(get_experiment_folder(cfg), 'domains.yaml'), 'w') as f:
            yaml.dump(cfg['domains'], f, default_flow_style=False)

        with open(os.path.join(get_experiment_folder(cfg), experiment_name + '_config.yaml'), 'w') as f:
            yaml.dump(temp_cfg, f, default_flow_style=False)
            
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
