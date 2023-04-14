from avalanche.training.plugins import EvaluationPlugin
from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate, BaseTemplate
from typing import Sequence
import torch

import strategies.cotta as cotta
from strategies.frozen_strategy import FrozenModel


def get_cotta_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = torch.optim.Adam(params,
                                 lr=cfg['lr'],
                                 betas=(cfg['beta'], 0.999),
                                 weight_decay=cfg['weight_decay'])
    cotted_model = cotta.CoTTA(model, optimizer,
                       steps=cfg['steps'],
                       episodic=cfg['episodic'],
                       mt_alpha=cfg['mt'],
                       rst_m=cfg['rst'],
                       ap=cfg['ap'],
                       img_size=cfg['img_size'])
    plugins.append(CottaPlugin())

    return FrozenModel(
        cotted_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    
    
class CottaPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.adapt = True

    def before_eval(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.adapt = False
