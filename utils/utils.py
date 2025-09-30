import os
import random
import subprocess

import numpy as np
import torch
import yaml


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
    return os.path.join(get_experiment_folder(cfg), 'seed' + str(cfg['seed']))

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def save_config(cfg):
    folder = get_seed_folder(cfg)
    os.makedirs(folder, exist_ok = True)
    with open(os.path.join(folder, get_experiment_name(cfg) + '_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

def get_device(cfg):
    return torch.device(f"cuda:{cfg['cuda']}"
                if torch.cuda.is_available() and cfg['cuda'] >= 0
                else "cpu")
