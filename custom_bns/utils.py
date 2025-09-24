from torch import nn


def replace_bn(model: nn.Module, BN_module: nn.Module, **abn_kwargs):
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
            n_replaced += replace_bn(
                target_mod, BN_module, **abn_kwargs)
    return n_replaced

def count_bn(model: nn.Module, BN_module):
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, BN_module):
            cnt += 1
    return cnt
