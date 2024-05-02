from avalanche.training.plugins import EvaluationPlugin
from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch

import strategies.rmt as rmt
from strategies.frozen_strategy import FrozenModel


def get_rmt_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = rmt.configure_model(model)
    params, param_names = rmt.collect_params(model)

    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,
                                    lr=cfg['lr'],
                                    betas=(cfg['beta'], 0.999),
                                    weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    cotted_model = rmt.RMT(model, optimizer, cfg,
                       steps=cfg['steps'],
                       episodic=cfg['episodic'],
                       img_size=cfg['img_size'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        cotted_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

