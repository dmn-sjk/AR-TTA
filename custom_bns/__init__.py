import torch

from .utils import replace_bn, count_bn
from .dynamic_bn import DynamicBN
from .mecta_bn import MectaBN
from .mecta_bn_original import MectaNorm2d as MectaBNOrig
from .AdaMixBN import AdaMixBN
# from .old_dynamic_bn import DynamicBN

def configure_model_bn(cfg, model):
    if cfg['bn_stats'] == 'source':
        print("Using source BN statistics!")
        return model
    elif cfg['bn_stats'] == 'test':
        print("Using test BN statistics!")
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        return model
    elif cfg['bn_stats'] == 'dynamicbn':
        BN_to_inject = DynamicBN
    elif cfg['bn_stats'] == 'mectabn':
        BN_to_inject = MectaBN
    elif cfg['bn_stats'] == 'mectabn_original':
        BN_to_inject = MectaBNOrig
    elif cfg['bn_stats'] == 'adamixbn':
        BN_to_inject = AdaMixBN
    else:
        raise ValueError(f"No such bn stats method: {cfg['bn_stats']}")
        
    print(f"Using {BN_to_inject.__name__} BN statistics!")
    
    n_bn = count_bn(model, torch.nn.BatchNorm2d)
    n_bn_to_replace = int(n_bn * cfg['fraction_bn_mod'])
    
    n_repalced = replace_bn(model, BN_to_inject,
                            number_to_replace=n_bn_to_replace,
                            beta=cfg['init_beta'],
                            bn_dist_scale=cfg['bn_dist_scale'],
                            smoothing_beta=cfg['smoothing_beta'],
                            use_forget_gate=True,
                            beta_thre=0., # 0.00125 for cifar and imagenet
                            prune_q=0., # 0.7 for all
                            transform=False
                   )
    assert n_repalced == n_bn_to_replace, f"Replaced {n_repalced} BNs but you wanted to replace {n_bn_to_replace}. Need to update `replace_bn`."

    n_bn_inside = count_bn(model, BN_to_inject)
    assert n_repalced == n_bn_inside, f"Replaced {n_repalced} BNs but actually inserted {n_bn_inside} {BN_to_inject.__name__}."
    
    return model