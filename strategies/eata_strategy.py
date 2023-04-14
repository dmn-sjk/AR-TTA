from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence
import torch

import strategies.eata as eata
from strategies.frozen_strategy import FrozenModel

# TODO:

def get_eata_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    return 
    fisher_dataset, fisher_loader = prepare_test_data(args)
    fisher_dataset.set_dataset_size(cfg['fisher_size'])
    fisher_dataset.switch_mode(True, False)

    subnet = eata.configure_model(subnet)
    params, param_names = eata.collect_params(subnet)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
    for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
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
    print("compute fisher matrices finished")
    del ewc_optimizer

    optimizer = torch.optim.SGD(params, cfg['lr'], momentum=0.9)
    adapt_model = eata.EATA(subnet, optimizer, fishers, cfg['fisher_alpha'], e_margin=cfg['e_margin'], d_margin=cfg['d_margin'])

    return FrozenModel(
        adapt_model, train_mb_size=cfg['batch_size'], eval_mb_size=32,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
