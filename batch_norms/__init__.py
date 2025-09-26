from torch import nn


def replace_bns(cfg, model, BN_to_inject: nn.BatchNorm2d):
    # if cfg['bn_stats'] == 'source':
    #     print("Using source BN statistics!")
    #     return model
    # elif cfg['bn_stats'] == 'test':
    #     print("Using test BN statistics!")
    #     for m in model.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             # force use of batch stats in train and eval modes
    #             m.track_running_stats = False
    #             m.running_mean = None
    #             m.running_var = None
    #     return model
    # elif cfg['bn_stats'] == 'dynamicbn':
    #     BN_to_inject = DynamicBN
    # else:
    #     raise ValueError(f"No such bn stats method: {cfg['bn_stats']}")
        
    print(f"Using {BN_to_inject.__name__} BN statistics!")
    
    n_bn = _count_bn(model, nn.BatchNorm2d)
    
    n_replaced = _replace_bn(model, BN_to_inject,
                            beta=cfg['init_beta'],
                            bn_dist_scale=cfg['bn_dist_scale'],
                            smoothing_beta=cfg['smoothing_beta'],
                            use_forget_gate=True,
                            beta_thre=0., # 0.00125 for cifar and imagenet
                            prune_q=0., # 0.7 for all
                            transform=False
                   )
    assert n_replaced == n_bn, f"Replaced {n_replaced} BNs but you wanted to replace {n_bn}. Need to update `replace_bn`."

    n_bn_inside = _count_bn(model, BN_to_inject)
    assert n_replaced == n_bn_inside, f"Replaced {n_replaced} BNs but actually inserted {n_bn_inside} {BN_to_inject.__name__}."
    
    return model

def _replace_bn(model: nn.Module, BN_module: nn.Module, **abn_kwargs):
    copy_keys = ['eps', 'momentum', 'affine']
    n_replaced = 0
    for mod_name, target_mod in model.named_children():
        
        if isinstance(target_mod, nn.BatchNorm2d) or isinstance(target_mod, nn.SyncBatchNorm):
            n_replaced += 1
            
            new_mod = BN_module(
                target_mod.num_features,
                **{k: getattr(target_mod, k) for k in copy_keys},
                **abn_kwargs,
            )
            new_mod.load_state_dict(target_mod.state_dict())
            new_mod.track_running_stats = False
            setattr(model, mod_name, new_mod)
        else:
            n_replaced += _replace_bn(
                target_mod, BN_module, **abn_kwargs)
    return n_replaced

def _count_bn(model: nn.Module, BN_module):
    cnt = 0
    for _, m in model.named_modules():
        if isinstance(m, BN_module):
            cnt += 1
    return cnt