import clad
from utils.transforms import get_transforms
from typing import Sequence
from avalanche.benchmarks.scenarios import GenericCLScenario


def get_cladc_benchmark(cfg) -> Sequence[GenericCLScenario, Sequence[str]]:
    transforms_test = get_transforms(cfg, train=False)
    
    # TODO: add 1st experience as source stream for eval
    benchmark = clad.cladc_avalanche(cfg["data_root"], test_transform=transforms_test, train_trasform=transforms_test, img_size=cfg["img_size"])
    cfg['domains'] = [experience.task_label for experience in benchmark.test_stream]
    return benchmark

    