from torch import nn

from utils.adapt_turnoff_plugin import AdaptTurnoffPlugin
import strategies.tent as tent
from strategies.frozen_strategy import FrozenModel
from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence
from . import register_strategy
from utils.optim import get_optimizer


@register_strategy("tent")
def get_tent_strategy(cfg, model: nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)

    optimizer = get_optimizer(cfg, params)
 
    tented_model = tent.Tent(model, optimizer)
    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        tented_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
