from avalanche.training import Naive
import torchvision
import torch
from avalanche.training import Naive

from .frozen_strategy import get_frozen_strategy
from .tent_strategy import get_tent_strategy
from .cotta_strategy import get_cotta_strategy
from loggers import get_eval_plugin


def get_strategy(cfg):
    if cfg['model'] == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif cfg['model'] == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model name: {cfg['model']}")

    model.fc = torch.nn.Linear(model.fc.in_features, cfg['num_classes'], bias=True)

    if cfg['pretrained_model_path'] is not None:
        model.load_state_dict(torch.load(cfg['pretrained_model_path']))
        
    model.to(cfg['device'])

    if cfg['watch_model']:
        eval_plugin = get_eval_plugin(cfg, model)
    else:
        eval_plugin = get_eval_plugin(cfg)

    plugins = []
    if cfg['method'] == "finetune":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'])
        criterion = torch.nn.CrossEntropyLoss()
        strategy = Naive(
            model, optimizer, criterion, train_mb_size=cfg['batch_size'], train_epochs=1, eval_mb_size=128,
            device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    elif cfg['method'] == "frozen":
        strategy = get_frozen_strategy(cfg, model, eval_plugin, plugins)
    elif cfg['method'] == "tent":
        strategy = get_tent_strategy(cfg, model, eval_plugin, plugins)
    elif cfg['method'] == "cotta":
        strategy = get_cotta_strategy(cfg, model, eval_plugin, plugins)
    elif cfg['method'] == "eata":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown method: {cfg['method']}")
    
    return strategy