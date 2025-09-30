from torch import nn

_METHODS = {}

def register_strategy(name):
    def decorator(func):
        _METHODS[name] = func
        return func
    return decorator

def get_method(cfg: dict, model: nn.Module):
    try:
        return _METHODS[cfg['method']](cfg, model)
    except KeyError:
        raise ValueError(f"Unknown strategy: {cfg['method']}")


from . import source, artta, rmt, eata, sar, tent, bn_1, cotta
