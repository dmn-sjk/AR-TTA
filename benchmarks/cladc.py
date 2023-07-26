import clad
from utils.transforms import get_transforms
from avalanche.benchmarks.scenarios import GenericCLScenario
import numpy as np


def get_cladc_benchmark(cfg) -> GenericCLScenario:
    transforms_test = get_transforms(cfg, train=False)
    
    # TODO: add 1st experience as source stream for eval
    benchmark = clad.cladc_avalanche(cfg["data_root"], test_transform=transforms_test, train_trasform=transforms_test, img_size=cfg["img_size"],
                                     end_with_source_domain=cfg["end_with_source_domain"])

    cfg['domains'] = [f"T{experience.task_label + 1}" for experience in benchmark.train_stream]
    if cfg["end_with_source_domain"]:
        cfg['domains'][-1] = 'T0'


    if 'random' in cfg['benchmark']:
        if cfg['benchmark'] == 'clad_long_random':
            replace = True
            size = 150
        else:
            replace = False
            size = 5

        if cfg["end_with_source_domain"]:
            cfg['domains'] = np.random.choice(cfg['domains'][:-1], size=size, replace=replace).astype(str).tolist()
            cfg['domains'].append('T0')
        else:
            cfg['domains'] = np.random.choice(cfg['domains'], size=size, replace=replace).astype(str).tolist()

    return benchmark

def domain_to_experience_idx(domain):
    task_nr = int(domain[-1])
    if task_nr == 0:
        return 5
    return task_nr - 1
    