from avalanche.training.plugins import EvaluationPlugin
from utils.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch
import numpy as np

from strategies import artta
from strategies.frozen_strategy import FrozenModel
from datasets.cifar10c import CIFAR10CDataset
from datasets.imagenetc import ImageNetCDataset
from datasets.shift import SHIFTClassificationDataset
from datasets.shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from datasets.shift_dev.utils.backend import ZipBackend
from datasets import clad
from utils.optim import get_optimizer
from . import register_strategy



@register_strategy("artta")
def get_artta_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    model = artta.configure_model(model)
    params, param_names = artta.collect_params(model)

    optimizer = get_optimizer(cfg, params)

    memory = None
    if cfg['memory_size'] != 0:
        if cfg['dataset'] == 'cifar10c' or cfg['dataset'] == 'cifar10_1':
            train_dataset = CIFAR10CDataset(cfg['data_root'], corruption=None, split='train', transforms=None)
        elif cfg['dataset'] == 'shift':
            train_dataset = SHIFTClassificationDataset(split='train',
                                                        data_root=cfg['data_root'],
                                                        transforms=None,
                                                        weathers_coarse=[WeathersCoarse.clear],
                                                        timeofdays_coarse=[
                                                            TimesOfDayCoarse.daytime],
                                                        backend=ZipBackend(),
                                                        classification_img_size=cfg['img_size'])
        elif cfg['dataset'] == 'clad':
            train_dataset = clad.get_cladc_train(cfg['data_root'], transform=None, sequence_type='source',
                                                 img_size=cfg['img_size'])[0]
        elif cfg['dataset'] == 'imagenetc':
            train_dataset = ImageNetCDataset(cfg['data_root'], corruption=None, split="train", transform=None,
                                     img_size=cfg['img_size'])
        else:
            raise NotImplementedError

        memory = {'x': torch.Tensor(), 'labels': torch.LongTensor()}

        # class-balanced memory
        for class_id in range(cfg['num_classes']):
            class_idxs = (torch.Tensor(train_dataset.targets) == class_id).nonzero(as_tuple=True)[0]
            
            if 'memory_size' in cfg.keys():
                memory_per_class = int(cfg['memory_size'] / cfg['num_classes'])

                rest = cfg['memory_size'] % cfg['num_classes']
                if class_id < rest:
                    chosen_idxs = np.random.choice(class_idxs, cfg['memory_size'] // cfg['num_classes'] + 1)
                else:
                    chosen_idxs = np.random.choice(class_idxs, cfg['memory_size'] // cfg['num_classes'])
                    
            else:
                chosen_idxs = np.random.choice(class_idxs, memory_per_class)

            for idx in chosen_idxs:
                # train_dataset[idx][0] - single sample image, adding dimension with None 
                memory['x'] = torch.cat((memory['x'], train_dataset[idx][0][None,:]), dim=0)
                memory['labels'] = torch.cat((memory['labels'], torch.LongTensor([train_dataset[idx][1]])), dim=0)
                
        random_order_idxs = torch.randperm(len(memory['labels']))
        memory['x'] = memory['x'][random_order_idxs]
        memory['labels'] = memory['labels'][random_order_idxs]

    artta_model = artta.ARTTA(model, optimizer, cfg,
                       steps=cfg['steps'],
                       img_size=cfg['img_size'],
                       memory=memory,
                       alpha=cfg['alpha'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        artta_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

