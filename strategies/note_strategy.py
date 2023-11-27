import torch
import numpy as np
from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence

from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from strategies.frozen_strategy import FrozenModel
from strategies import note
from custom_bns import iabn

def get_note_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    iabn.convert_iabn(model, k=cfg['iabn_k'])

    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=cfg['lr'],
                                    weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    model = note.configure_model(model, cfg['bn_momentum'])

    note_model = note.NOTE(model, optimizer, memory_size=cfg['memory_size'],
                           num_classes=cfg['num_classes'])
    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        note_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)