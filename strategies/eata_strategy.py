from avalanche.training.plugins import EvaluationPlugin
from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from typing import Sequence
import torch
from copy import copy
import numpy as np

import strategies.eata as eata
from strategies.eata import softmax_entropy
from strategies.frozen_strategy import FrozenModel
from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from benchmarks.cifar10c import CIFAR10CDataset
from benchmarks.shift import SHIFTClassificationDataset
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from datasets.imagenetc import ImageNetCDataset
from shift_dev.utils.backend import ZipBackend
import clad


def get_eata_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    fisher_batch_size = 64
    if cfg['dataset'] == 'cifar10c':
        fisher_dataset = CIFAR10CDataset(cfg['data_root'], corruption=None, split='test', transforms=None)
    elif cfg['dataset'] == 'shift':
        fisher_dataset = SHIFTClassificationDataset(split='val',
                                                    data_root=cfg['data_root'],
                                                    transforms=None,
                                                    weathers_coarse=[WeathersCoarse.clear],
                                                    timeofdays_coarse=[
                                                        TimesOfDayCoarse.daytime],
                                                    backend=ZipBackend(),
                                                    classification_img_size=cfg['img_size'])
    elif cfg['dataset'] == 'clad':
        # TODO: for now val set has all the domains, maybe modify for only daytime and depending on the possibilities match the weather with train set 
        fisher_dataset = clad.get_cladc_val(cfg['data_root'], transform=None)
    elif cfg['dataset'] == 'imagenetc':
        fisher_dataset = ImageNetCDataset(cfg['data_root'], corruption=None, split='val', transform=None,
                                     img_size=cfg['img_size'])
    else:
        raise NotImplementedError

    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=fisher_batch_size, shuffle=True, 
                                                    num_workers=cfg['num_workers'], pin_memory=True)

    subnet = eata.configure_model(model)
    params, param_names = eata.collect_params(subnet)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
    stop = False
    for iter_, (images, targets) in enumerate(fisher_loader, start=1):
        
        num_seen_samples_after_iter = iter_ * fisher_batch_size
        if num_seen_samples_after_iter > cfg['fisher_size']:
            num_samples_to_use = fisher_batch_size - (num_seen_samples_after_iter - cfg['fisher_size'])
            images = images[:num_samples_to_use]
            targets = targets[:num_samples_to_use]
            stop = True
            
        if cfg['cuda'] is not None:
            images = images.cuda(cfg['cuda'], non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(cfg['cuda'], non_blocking=True)
        outputs = subnet(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in subnet.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()

        if stop:
            break

    print("compute fisher matrices finished")
    del ewc_optimizer
    del fisher_loader

    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,
                                    lr=cfg['lr'],
                                    betas=(cfg['beta'], 0.999),
                                    weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    cfg['e_margin'] = float(cfg['e_margin_coeff'] * np.log(cfg['num_classes']))

    adapt_model = eata.EATA(subnet, optimizer, fishers, cfg['fisher_alpha'], e_margin=cfg['e_margin'], d_margin=cfg['d_margin'])
    
    plugins.append(AdaptTurnoffPlugin())
    
    if cfg['fisher_online']:
        plugins.append(OnlineFisherPlugin(fishers, cfg['fisher_update_every'], cfg['e_margin']))

    return FrozenModel(
        adapt_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)


class OnlineFisherPlugin(SupervisedPlugin):
    def __init__(self, current_fishers, update_every, e_margin = 0.921034037, gamma = 0.95):
        super().__init__()
        self.current_fishers = copy(current_fishers)
        self.update_every = update_every
        self.sample_count = 0
        self.fishers = {}
        self.e_margin = e_margin
        self.gamma = gamma
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        
    @torch.enable_grad()
    def after_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
        super().after_training_iteration(strategy, **kwargs)

        # current models outputs
        # outputs = strategy.mb_output
        
        strategy.model.optimizer.zero_grad()
        
        # TODO: instead get outpus from mb_output to avoid double forward (for now problems with grad)
        inputs = strategy.mb_x
        # get outputs directly from the model, not EATA
        outputs = strategy.model.model(inputs)
         
        entropys = softmax_entropy(outputs)
        # filter unreliable samples
        filter_ids = torch.where(entropys < self.e_margin)[0]
        
        if len(filter_ids) == 0:
            return
        
        outputs = outputs[filter_ids]
        _, targets = outputs.max(1)


        # TODO: correct with exp_cond_prob
        raise NotImplementedError
        # loss = - torch.nn.functional.nll_loss(self.logsoft(outputs), targets,
        #                             reduction='none')
        # exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
        # loss = torch.mean(loss)


        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        loss.backward()

        self.sample_count += len(outputs)

        for name, param in strategy.model.named_parameters():
            # remove 'model.' from the beginning of name
            name = name[6:]
            if param.grad is not None:
                if name in self.fishers.keys():
                    # fisher = exp_cond_prob * param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
                    fisher = param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
                else:
                    # fisher = exp_cond_prob * param.grad.data.clone().detach() ** 2
                    fisher = param.grad.data.clone().detach() ** 2
                if self.sample_count >= self.update_every:
                    fisher = fisher / self.sample_count
                self.fishers.update({name: [fisher, param.data.clone().detach()]})
        
        if self.sample_count >= self.update_every:
            self.sample_count = 0
            
            for name in self.current_fishers.keys():
                self.current_fishers[name][0] *= self.gamma
                self.current_fishers[name][0] += self.fishers[name][0]

            strategy.model.fishers = copy(self.current_fishers)
            
            self.fishers = {}
