import clad
from utils.transforms import get_transforms
from avalanche.benchmarks.scenarios import GenericCLScenario


def get_cladc_benchmark(cfg) -> GenericCLScenario:
    transforms_test = get_transforms(cfg, train=False)
    
    # TODO: add 1st experience as source stream for eval
    benchmark = clad.cladc_avalanche(cfg["data_root"], test_transform=transforms_test, train_trasform=transforms_test, img_size=cfg["img_size"],
                                     end_with_source_domain=cfg["end_with_source_domain"])

    cfg['domains'] = [f"T{experience.task_label + 1}" for experience in benchmark.train_stream]
    if cfg["end_with_source_domain"]:
        cfg['domains'][-1] = 'T0'

    return benchmark

    