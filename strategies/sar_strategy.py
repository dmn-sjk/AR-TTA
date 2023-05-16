import torch
import numpy as np
from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence

from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from strategies import sar
from strategies.frozen_strategy import FrozenModel
from utils.sam import SAM


def get_sar_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    
    if cfg['optimizer'] == 'adam':
        base_optimizer = torch.optim.Adam
        optimizer = SAM(params, base_optimizer, lr=cfg['lr'], betas=(cfg['beta'], 0.999), weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
    
    cfg['margin_e0'] = float(0.4 * np.log(cfg['num_classes']))

    sar_model = sar.SAR(model, optimizer, margin_e0=cfg['margin_e0'])
    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        sar_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
