from avalanche.training.plugins import EvaluationPlugin
from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch
import numpy as np

from strategies.nonparametric import Nonparametric
from strategies.frozen_strategy import FrozenModel
from torch.nn import functional as F
from functools import partial


def get_nonparametric_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):

    model = Nonparametric(model, cfg)

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

