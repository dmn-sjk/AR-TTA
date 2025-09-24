from torch import nn
import torch
import random
import numpy as np
import os
import subprocess
import yaml
from copy import copy, deepcopy


# ref: https://github.com/Oldpan/Pytorch-Memory-Utils/blob/master/gpu_mem_track.py
dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}

def norm_params_unchanged(strategy, prev_norm_params):
        for m in strategy.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                if prev_norm_params['running_mean'] is not None:
                    if not torch.all(torch.eq(prev_norm_params['running_mean'], m.running_mean.cpu())):
                        print('running_mean NOT THE SAME')

                if prev_norm_params['running_var'] is not None:
                    if not torch.all(torch.eq(prev_norm_params['running_var'], m.running_var.cpu())):
                        print('running_var NOT THE SAME')

                prev_norm_params['running_mean'] = m.running_mean.cpu()
                prev_norm_params['running_var'] = m.running_var.cpu()

                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        if prev_norm_params['named'] is not None:
                            if not torch.all(torch.eq(prev_norm_params['named'], p.data.cpu())):
                                print('named NOT THE SAME')

                        prev_norm_params['named'] = p.data.cpu()
                        break
                break

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_experiment_name(cfg: dict) -> str:
    return f"{cfg['dataset']}_{cfg['method']}_{cfg['model']}_{cfg['run_name']}"

def get_experiment_folder(cfg: dict) -> str:
    return os.path.join(cfg['log_dir'], get_experiment_name(cfg))

def get_seed_folder(cfg: dict) -> str:
    return os.path.join(get_experiment_folder(cfg), 'seed' + str(cfg['curr_seed']))

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def split_up_model(model, model_name, dataset=None, encoder_out_relu_to_classifier=False):
    """
    encoder_out_relu_to_classifier: encoder has relu activation at the end, if this argument is True
        this relu is moved to classifier (works for wideresnet only now)
        TODO: check sensibility of doing that for resnet since it has relu on output inside one of the bottlenecks
    """
    if encoder_out_relu_to_classifier and 'wideresnet' not in model_name:
        raise NotImplementedError()
    
    if 'wideresnet' in model_name: 
        if encoder_out_relu_to_classifier:
            encoder = nn.Sequential(*list(model.children())[:-2], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        else:
            encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
    elif 'resnet' in model_name:
        if dataset == 'imagenetc':
            encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
            model = model.model
        else:
            encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        
    if encoder_out_relu_to_classifier and 'wideresnet' in model_name:
        classifier = nn.Sequential(list(model.children())[-2], model.fc)
    else:
        classifier = model.fc

    return encoder, classifier

# TODO: merge with split_up_model
def split_up_model_new(model, model_name, dataset_name: str):
    if "wideresnet" in model_name and dataset_name in {"cifar10", "cifar10c", "cifar10_1"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
    elif 'resnet' in model_name:
        if dataset_name == 'imagenetc':
            encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
            model = model.model
        else:
            encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    else:
        raise NotImplementedError()
    
    classifier = model.fc
    return encoder, classifier
    
    
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    Input:
        model: Model to be split up
        arch_name: Name of the network
        dataset_name: Name of the dataset
    Returns:
        encoder: The encoder of the model
        classifier The classifier of the model
    """
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        # split up models loaded from timm
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)

    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c", "cifar10_1"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50", "Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    # add a masking layer to the classifier
    if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        mask = eval(f"{dataset_name.upper()}_MASK")
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))

    return encoder, classifier

def save_config(cfg, experiment_name):
    temp_cfg = copy(cfg)
    del temp_cfg['domains']
    
    if cfg['curr_seed'] is not None:
        with open(os.path.join(get_seed_folder(cfg), 'domains.yaml'), 'w') as f:
            yaml.dump(cfg['domains'], f, default_flow_style=False)

        del temp_cfg['curr_seed']
        with open(os.path.join(get_experiment_folder(cfg), experiment_name + '_config.yaml'), 'w') as f:
            yaml.dump(temp_cfg, f, default_flow_style=False)
    else:
        with open(os.path.join(get_experiment_folder(cfg), 'domains.yaml'), 'w') as f:
            yaml.dump(cfg['domains'], f, default_flow_style=False)

        with open(os.path.join(get_experiment_folder(cfg), experiment_name + '_config.yaml'), 'w') as f:
            yaml.dump(temp_cfg, f, default_flow_style=False)
            
def gauss_symm_kl_divergence(mean1, var1, mean2, var2, eps):
    # >>> out-place ops
    dif_mean = (mean1 - mean2) ** 2
    d1 = var1 + eps + dif_mean
    d1.div_(var2 + eps)
    d2 = (var2 + eps + dif_mean)
    d2.div_(var1 + eps)
    d1.add_(d2)
    d1.div_(2.).sub_(1.)
    # d1 = (var1 + eps + dif_mean) / (var2 + eps) + (var2 + eps + dif_mean) / (var1 + eps)
    return d1
