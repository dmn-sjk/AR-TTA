from avalanche.training.plugins import EvaluationPlugin
from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch

from strategies import custom
from strategies.frozen_strategy import FrozenModel


def get_cotta_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = custom.configure_model(model)
    params, param_names = custom.collect_params(model)

    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,
                                    lr=cfg['lr'],
                                    betas=(cfg['beta'], 0.999),
                                    weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    cotted_model = custom.Custom(model, optimizer,
                       steps=cfg['steps'],
                       img_size=cfg['img_size'],
                       distillation_out_temp=cfg['distillation_out_temp'],
                       features_distillation_weight=cfg['features_distillation_weight'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        cotted_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

