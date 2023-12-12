from torch import nn

from strategies.frozen_strategy import FrozenModel
from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence


def get_bn_stats_adapt_strategy(cfg, model: nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    # now it is handled before this function so this specific stategy is actually useless now
    # model = configure_model(model)

    return FrozenModel(
        model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
