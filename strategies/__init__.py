from avalanche.training import Naive
import torchvision
import torch
from avalanche.training import Naive
from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive
from robustbench.utils import load_model
import timm

from .frozen_strategy import get_frozen_strategy
from .tent_strategy import get_tent_strategy
from .cotta_strategy import get_cotta_strategy
from .eata_strategy import get_eata_strategy
from .note_strategy import get_note_strategy
from .sar_strategy import get_sar_strategy
from .custom_strategy import get_custom_strategy
from .bn_stats_adapt_strategy import get_bn_stats_adapt_strategy
from .ema_teacher_strategy import get_ema_teacher_strategy
from loggers import get_eval_plugin


def get_strategy(cfg):
    if cfg['model'] == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif cfg['model'] == 'resnet50':
        if cfg['dataset'] == 'imagenetc':
            # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            model = load_model('Standard_R50', cfg['model_ckpt_dir'],
                            'imagenet', "corruptions")
        else:
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    elif cfg['model'] == 'wideresnet28':
        # if 'cifar10' not in cfg['dataset']
            # raise ValueError(f"Robust bench wideresnet28 pretrained model only available for cifar10 dataset")

        # if cfg['dataset'] == 'cifar10c':
        #     dataset = 'cifar10'
        # else:
        #     dataset = cfg['dataset']

        model = load_model('Standard', cfg['model_ckpt_dir'],
                            'cifar10', "corruptions")
    elif cfg['model'] == 'resnet50gn':
        model = timm.create_model('resnet50_gn', pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {cfg['model']}")

    if not (cfg['dataset'] == 'cifar10c' and cfg['model'] == 'wideresnet28' \
        or cfg['dataset'] == 'imagenetc' and cfg['model'] == 'resnet50'):    
        model.fc = torch.nn.Linear(model.fc.in_features, cfg['num_classes'], bias=True)

    if 'pretrained_model_path' in cfg.keys():
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
        strategy = get_eata_strategy(cfg, model, eval_plugin, plugins)
    elif cfg['method'] == 'note':
        strategy = get_note_strategy(cfg, model, eval_plugin, plugins)
    elif cfg['method'] == 'sar':
        strategy = get_sar_strategy(cfg, model, eval_plugin, plugins)
    elif cfg['method'] == 'custom':
        strategy = get_custom_strategy(cfg, model, eval_plugin, plugins)
    elif cfg['method'] == 'bn_stats_adapt':
        strategy = get_bn_stats_adapt_strategy(cfg, model, eval_plugin, plugins)
    elif cfg['method'] == 'ema_teacher':
        strategy = get_ema_teacher_strategy(cfg, model, eval_plugin, plugins)
    else:
        raise ValueError(f"Unknown method: {cfg['method']}")
    
    return strategy