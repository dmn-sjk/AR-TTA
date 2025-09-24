from datasets import clad
from utils.transforms import get_transforms
from . import register_benchmark

from avalanche.benchmarks.scenarios import GenericCLScenario


@register_benchmark("clad")
def get_cladc_benchmark(cfg) -> GenericCLScenario:
    transforms_test = get_transforms(cfg, train=False)
    
    benchmark = clad.cladc_avalanche(cfg["data_root"], test_transform=transforms_test, train_trasform=transforms_test, img_size=cfg["img_size"])

    cfg['domains'] = [f"T{experience.task_label + 1}" for experience in benchmark.train_stream]

    return benchmark

def domain_to_experience_idx(domain):
    task_nr = int(domain[-1])
    if task_nr == 0:
        return 5
    return task_nr - 1
    