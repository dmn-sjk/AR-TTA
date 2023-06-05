from avalanche.training.plugins import EvaluationPlugin
from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch
import numpy as np

from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.storage_policy import ClassBalancedBuffer
from types import SimpleNamespace
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

from strategies import custom
from strategies.frozen_strategy import FrozenModel
from benchmarks.cifar10c import CIFAR10CDataset
from benchmarks.shift import SHIFTClassificationDataset
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from shift_dev.utils.backend import ZipBackend
import clad



def get_custom_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
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

    memory = None
    if cfg['exemplars']:
        if cfg['dataset'] == 'cifar10c':
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
            # TODO: for now val set has all the domains, maybe modify for only daytime and depending on the possibilities match the weather with train set 
            train_dataset = clad.get_cladc_train(cfg['data_root'], transform=None, sequence_type='source')[0]
        else:
            raise NotImplementedError

        memory = {'x': torch.Tensor(), 'labels': torch.LongTensor()}

        # class-balanced memory
        for class_id in range(cfg['num_classes']):
            class_idxs = (torch.Tensor(train_dataset.targets) == class_id).nonzero(as_tuple=True)[0]
            chosen_idxs = np.random.choice(class_idxs, cfg['memory_per_class'])

            for idx in chosen_idxs:
                # train_dataset[idx][0] - single sample image, adding dimension with None 
                memory['x'] = torch.cat((memory['x'], train_dataset[idx][0][None,:]), dim=0)
                memory['labels'] = torch.cat((memory['labels'], torch.LongTensor([train_dataset[idx][1]])), dim=0)
            
        random_order_idxs = torch.randperm(len(memory['labels']))
        memory['x'] = memory['x'][random_order_idxs]
        memory['labels'] = memory['labels'][random_order_idxs]

    custom_model = custom.Custom(model, optimizer, cfg,
                       steps=cfg['steps'],
                       img_size=cfg['img_size'],
                       distillation_out_temp=cfg['distillation_out_temp'],
                       features_distillation_weight=cfg['features_distillation_weight'],
                       memory=memory,
                       num_replay_samples=cfg['num_replay_samples'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        custom_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

