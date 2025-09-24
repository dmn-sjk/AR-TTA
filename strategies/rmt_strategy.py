from avalanche.training.plugins import EvaluationPlugin
from utils.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch

import strategies.rmt as rmt
from strategies.frozen_strategy import FrozenModel
from . import register_strategy
from utils.optim import get_optimizer


@register_strategy("rmt")
def get_rmt_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = rmt.configure_model(model)
    params, param_names = rmt.collect_params(model)
    
    optimizer = get_optimizer(cfg, params)

    cotted_model = rmt.RMT(model, optimizer, cfg,
                       steps=cfg['steps'],
                       episodic=cfg['episodic'],
                       img_size=cfg['img_size'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        cotted_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

