from torch import nn

from strategies.frozen_strategy import FrozenModel
from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence


def get_bn_stats_adapt_strategy(cfg, model: nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = configure_model(model)

    return FrozenModel(
        model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)


def configure_model(model):
    """Configure model."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model
