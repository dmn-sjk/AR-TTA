_BENCHMARKS = {}

def register_benchmark(name):
    def decorator(func):
        _BENCHMARKS[name] = func
        return func
    return decorator

def get_benchmark(cfg: dict):
    benchmark_name = cfg["dataset"]
    try:
        return _BENCHMARKS[benchmark_name](cfg)
    except KeyError:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

from . import cifar10_1, cifar10c, cladc, imagenetc, shift
