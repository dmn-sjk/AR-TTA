import torch


def get_optimizer(cfg, params):
    if cfg['optimizer'] == 'adam':
        return torch.optim.Adam(params,
                                lr=cfg['lr'],
                                betas=(cfg['beta'], 0.999),
                                weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        return torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")