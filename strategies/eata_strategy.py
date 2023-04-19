from avalanche.training.plugins import EvaluationPlugin
from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate, BaseTemplate
from typing import Sequence
import torch

import strategies.eata as eata
from strategies.frozen_strategy import FrozenModel
from benchmarks.cifar10c import CIFAR10CDataset
from benchmarks.shift import SHIFTClassificationDataset
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from shift_dev.utils.backend import ZipBackend


def get_eata_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    fisher_batch_size = 64
    if cfg['dataset'] == 'cifar10c':
        fisher_dataset = CIFAR10CDataset(cfg['data_root'], corruption=None, split='test', transforms=None)
        fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=fisher_batch_size, shuffle=True, 
                                                    num_workers=cfg['num_workers'], pin_memory=True)
    elif cfg['dataset'] == 'shift':
        fisher_dataset = SHIFTClassificationDataset(split='val',
                                                    data_root=cfg['data_root'],
                                                    transforms=None,
                                                    weathers_coarse=[WeathersCoarse.clear],
                                                    timeofdays_coarse=[
                                                        TimesOfDayCoarse.daytime],
                                                    backend=ZipBackend(),
                                                    classification_img_size=cfg['img_size'])
        fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=fisher_batch_size, shuffle=True, 
                                                    num_workers=cfg['num_workers'], pin_memory=True)
    else:
        raise NotImplementedError

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

    optimizer = torch.optim.SGD(params, cfg['lr'], momentum=0.9)
    adapt_model = eata.EATA(subnet, optimizer, fishers, cfg['fisher_alpha'], e_margin=cfg['e_margin'], d_margin=cfg['d_margin'])
    
    plugins.append(EATAPlugin())

    return FrozenModel(
        adapt_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)


class EATAPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.adapt = True

    def before_eval(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.adapt = False


class OnlineEWCPlugin(SupervisedPlugin):
    def __init__(self):
        # TODO
        raise NotImplementedError
        super().__init__()
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None
        
    def after_update(self, strategy: "SupervisedTemplate", **kwargs):
        # current models outputs
        outputs = strategy.mb_output
        
        # current mini batch
        _, labels, _ = strategy.mbatch
        
        for output, label in zip (outputs, labels):
            strategy.model.optimizer.zero_grad()
            loss = - torch.nn.functional.nll_loss(self.logsoft(output), label.unsqueeze(0),
                                reduction='none')
            exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
            loss = torch.mean(loss)
            loss.backward()
            self.updated_fish += exp_cond_prob * self.net.get_grads() ** 2


            # for name, param in strategy.model.named_parameters():
            #     if param.grad is not None:
            #         if iter_ > 1:
            #             fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
            #         else:
            #             fisher = param.grad.data.clone().detach() ** 2
            #         if iter_ == len(fisher_loader):
            #             fisher = fisher / iter_
            #         fishers.update({name: [fisher, param.data.clone().detach()]})
            # ewc_optimizer.zero_grad()

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        self.updated_fish = torch.zeros_like(strategy.model.get_params())
        
    def after_training(self, strategy: "SupervisedTemplate", **kwargs):
        self.updated_fish /= (len(strategy.dataloader) * strategy.dataloader.batch_size)

        self.fish *= self.args.gamma
        self.fish += self.updated_fish