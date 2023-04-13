from torch import nn

from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence


def get_cotta_strategy(cfg, model: nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    raise NotImplementedError