from avalanche.training.plugins import EvaluationPlugin
from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate, BaseTemplate
from typing import Sequence
import torch

import strategies.eata as eata
from strategies.frozen_strategy import FrozenModel
from benchmarks.cifar10c import CIFAR10CDataset


def get_eata_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    fisher_batch_size = 64
    if cfg['dataset'] == 'cifar10c':
        fisher_dataset = CIFAR10CDataset(cfg['data_root'], corruption=None, split='test', transforms=None)
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
