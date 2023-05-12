import torch
from torch import nn

from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
import strategies.tent as tent
from strategies.frozen_strategy import FrozenModel
from avalanche.training.plugins import EvaluationPlugin
from typing import Sequence


def get_tent_strategy(cfg, model: nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    # model, params = get_tented_model_and_params(model)
    # optimizer = torch.optim.SGD(params, lr=1e-3)
    #
    # from tent import softmax_entropy
    #
    # def softmax_entropy_loss(x: torch.Tensor, _):
    #     return softmax_entropy(x).mean(0)
    #
    # criterion = softmax_entropy_loss
    #
    # plugins.append(TentPlugin())
    #
    # strategy = Naive(
    #     model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=128,
    #     device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)

    # plugins.append(TentPlugin(lr=1e-3))

    # ----
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    
    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,
                                    lr=cfg['lr'],
                                    betas=(cfg['beta'], 0.999),
                                    weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
        
    tented_model = tent.Tent(model, optimizer)
    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        tented_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)


# def get_tented_model_and_params(model):
#     model.requires_grad_(False)
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None

#     params = []
#     for nm, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm2d):
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)

#     return model, params
