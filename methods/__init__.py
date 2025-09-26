from utils.models import get_model

_METHODS = {}

def register_strategy(name):
    def decorator(func):
        _METHODS[name] = func
        return func
    return decorator

def get_method(cfg: dict):
    model = get_model(cfg)

    try:
        return _METHODS[cfg['method']](cfg, model)
    except KeyError:
        raise ValueError(f"Unknown strategy: {cfg['method']}")


from . import source, artta, rmt, eata, sar, tent, bn_1, cotta
