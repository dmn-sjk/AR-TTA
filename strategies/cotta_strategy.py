from avalanche.training.plugins import EvaluationPlugin
from utils.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch

import strategies.cotta as cotta
from strategies.frozen_strategy import FrozenModel
from . import register_strategy
from utils.optim import get_optimizer


@register_strategy("cotta")
def get_cotta_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    
    optimizer = get_optimizer(cfg, params)

    cotted_model = cotta.CoTTA(model, optimizer,
                       steps=cfg['steps'],
                       episodic=cfg['episodic'],
                       mt_alpha=cfg['mt'],
                       rst_m=cfg['rst'],
                       ap=cfg['ap'],
                       img_size=cfg['img_size'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        cotted_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

