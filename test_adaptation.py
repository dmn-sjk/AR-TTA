import torch
import os
from utils.utils import set_seed
from utils.config_parser import ConfigParser
from benchmarks import get_benchmark
from strategies import get_strategy
import json


def main():
    cfg = ConfigParser(mode="tta").get_config()

    cfg['device'] = torch.device(f"cuda:{cfg['cuda']}"
                                 if torch.cuda.is_available() and cfg['cuda'] >= 0
                                 else "cpu")

    if cfg['seed'] is not None:
        set_seed(cfg['seed'])

    benchmark = get_benchmark(cfg)
    strategy = get_strategy(cfg)

    for i, experience in enumerate(benchmark.train_stream):
        # strategy.train(experience, eval_streams=[benchmark.streams['val']], shuffle=False,
        #                num_workers=cfg['num_workers)

        if i == 0:
            print("Initial eval...")
            strategy.eval(benchmark.test_stream[0], num_workers=cfg['num_workers'])

        strategy.train(experience, eval_streams=[], shuffle=False,
                       num_workers=cfg['num_workers'])

        strategy.eval(benchmark.test_stream[0], num_workers=cfg['num_workers'])


if __name__ == '__main__':
    main()
