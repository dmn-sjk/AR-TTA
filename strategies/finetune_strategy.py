from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence
import torch
from avalanche.training import Naive

from . import register_strategy
from utils.optim import get_optimizer


@register_strategy("finetune")
def get_finetune_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    optimizer = get_optimizer(cfg, model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    return Naive(
        model, optimizer, criterion, train_mb_size=cfg['batch_size'], train_epochs=1, eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

