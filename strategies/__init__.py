from loggers import get_eval_plugin
from utils.models import get_model


_STRATEGIES = {}

def register_strategy(name):
    def decorator(func):
        _STRATEGIES[name] = func
        return func
    return decorator

def get_strategy(cfg: dict):
    model = get_model(cfg)
    eval_plugin = get_eval_plugin(cfg)
    plugins = []

    strategy_name = cfg["method"]

    try:
        return _STRATEGIES[strategy_name](cfg, model, eval_plugin, plugins)
    except KeyError:
        raise ValueError(f"Unknown strategy: {strategy_name}")


from . import artta_strategy, frozen_strategy, finetune_strategy, tent_strategy, cotta_strategy, eata_strategy, sar_strategy, \
    bn_stats_adapt_strategy, rmt_strategy
