from avalanche.training.plugins import EvaluationPlugin
from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch
import numpy as np

from strategies import ema_teacher
from strategies.frozen_strategy import FrozenModel
import clad
from torch.nn import functional as F



def get_ema_teacher_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    
    params_for_update = None

    model = ema_teacher.configure_model(model, 
                                   params_for_update=params_for_update, 
                                   num_first_blocks_for_update=cfg['num_first_blocks_for_update'])
    params, param_names = ema_teacher.collect_params(model)
    print(param_names)

    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,
                                    lr=cfg['lr'],
                                    betas=(cfg['beta'], 0.999),
                                    weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    custom_model = ema_teacher.EmaTeacher(model, optimizer, cfg,
                       steps=cfg['steps'],
                       img_size=cfg['img_size'],
                       distillation_out_temp=cfg['distillation_out_temp'],
                       features_distillation_weight=cfg['features_distillation_weight'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        custom_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

