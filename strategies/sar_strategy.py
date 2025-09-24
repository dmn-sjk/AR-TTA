import torch
import numpy as np
from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence

from utils.adapt_turnoff_plugin import AdaptTurnoffPlugin
from strategies import sar
from strategies.frozen_strategy import FrozenModel
from utils.sam import SAM
from . import register_strategy
from utils.optim import get_optimizer


@register_strategy("sar")
def get_sar_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    
    base_optimizer = get_optimizer(cfg, params)
    optimizer = SAM(base_optimizer)
    
    cfg['margin_e0'] = float(cfg['e_margin_coeff'] * np.log(cfg['num_classes']))

    sar_model = sar.SAR(model, optimizer, margin_e0=cfg['margin_e0'])
    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        sar_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
