from avalanche.training.plugins import EvaluationPlugin
from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch
import numpy as np

from strategies import ema_teacher
from strategies.frozen_strategy import FrozenModel
from benchmarks.cifar10c import CIFAR10CDataset
from benchmarks.imagenetc import ImageNetCDataset
from benchmarks.shift import SHIFTClassificationDataset
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from shift_dev.utils.backend import ZipBackend
import clad
from custom_bns.dynamic_bn import DynamicBN
from custom_bns.mecta_bn import MectaBN
# from custom_bns.old_dynamic_bn import DynamicBN
from custom_bns.utils import replace_bn, count_bn



def get_ema_teacher_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    
    params_for_update = None
    # /* dynamic BN
    BN_to_inject = DynamicBN
    
    n_bn = count_bn(model, torch.nn.BatchNorm2d)
    n_bn_to_replace = int(n_bn * 1.0)
    
    n_repalced = replace_bn(model, BN_to_inject,
                            number_to_replace=n_bn_to_replace,
                            beta=cfg['init_beta'],
                            bn_dist_scale=cfg['bn_dist_scale'],
                            smoothing_beta=cfg['smoothing_beta'],
                   )
    assert n_repalced == n_bn_to_replace, f"Replaced {n_repalced} BNs but you wanted to replace {n_bn_to_replace}. Need to update `replace_bn`."

    n_bn_inside = count_bn(model, BN_to_inject)
    assert n_repalced == n_bn_inside, f"Replaced {n_repalced} BNs but actually inserted {n_bn_inside} {BN_to_inject.__name__}."
    # dynamic BN */

    model = ema_teacher.configure_model(model, 
                                   params_for_update=params_for_update, 
                                   num_first_blocks_for_update=cfg['num_first_blocks_for_update'])
    params, param_names = ema_teacher.collect_params(model)


    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,
                                    lr=cfg['lr'],
                                    betas=(cfg['beta'], 0.999),
                                    weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    custom_model = ema_teacher.EmaTeacher(model, optimizer, cfg,
                       steps=cfg['steps'],
                       img_size=cfg['img_size'],
                       distillation_out_temp=cfg['distillation_out_temp'],
                       features_distillation_weight=cfg['features_distillation_weight'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        custom_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

