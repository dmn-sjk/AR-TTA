
import torchvision
import torch
from robustbench.utils import load_model
import timm


def get_model(cfg):
    if cfg['model'] == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif cfg['model'] == 'resnet50':
        if cfg['dataset'] == 'imagenetc':
            model = load_model('Standard_R50', cfg['model_ckpt_dir'],
                            'imagenet', "corruptions")
        else:
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    elif cfg['model'] == 'wideresnet28':
        if cfg['dataset'] == 'cifar10c':
            dataset = 'cifar10'
        else:
            dataset = cfg['dataset']

        model = load_model('Standard', cfg['model_ckpt_dir'],
                            dataset, "corruptions")
    elif cfg['model'] == 'resnet50gn':
        model = timm.create_model('resnet50_gn', pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {cfg['model']}")

    # TODO:
    if not (cfg['dataset'] in ['cifar10c', 'cifar10_1'] and cfg['model'] == 'wideresnet28' \
        or cfg['dataset'] == 'imagenetc' and cfg['model'] == 'resnet50'):    
        model.fc = torch.nn.Linear(model.fc.in_features, cfg['num_classes'], bias=True)

    if 'pretrained_model_path' in cfg.keys():
        model.load_state_dict(torch.load(cfg['pretrained_model_path']))
    # TODO: fix the pretrained_model_path and model_ckpt_dir args
    
    model = model.to(cfg['device'])
    model.eval()

    return model