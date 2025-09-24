from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
)
from datasets.shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from datasets.shift_dev.utils.backend import HDF5Backend, ZipBackend
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from avalanche.benchmarks.scenarios import GenericCLScenario

from utils.transforms import get_transforms
from datasets.shift import SHIFTClassificationDataset
from constants.shift import WEATHERS_SEQUENCE, TIMEOFDAY_SEQUENCE, STANDARD_SHIFT_MIX_DOMAIN_SEQUENCE
from . import register_benchmark


@register_benchmark("shift")
def get_shift_benchmark(cfg) -> GenericCLScenario:
    train_sets = []
    val_sets = []
    train_sets, val_sets, domains = _get_domains_mix_no_source_sets(cfg) 
    cfg['domains'] = domains

    train_exps_datasets = []
    for i, train_set in enumerate(train_sets):
        train_dataset_avl = make_classification_dataset(
            train_set,
            initial_transform_group="train",
            task_labels=i,
            
        )

        train_exps_datasets.append(
            classification_subset(train_dataset_avl)
        )

    val_exps_datasets = []
    for i, val_set in enumerate(val_sets):
        val_dataset_avl = make_classification_dataset(
            val_set,
            initial_transform_group="eval",
            task_labels=i
        )

        val_exps_datasets.append(
            classification_subset(val_dataset_avl)
        )

    return create_multi_dataset_generic_benchmark(train_datasets=train_exps_datasets,
                                                  test_datasets=val_exps_datasets,
                                                  # train_transform=None,
                                                  # eval_transform=None
                                                  )


def _get_domains_mix_no_source_sets(cfg):
    train_sets = []
    val_sets = []
    
    transforms_test = get_transforms(cfg, train=False)
    
    # source domain, but validation split
    val_sets.append(SHIFTClassificationDataset(split='val',
                                                data_root=cfg['data_root'],
                                                transforms=transforms_test,
                                                weathers_coarse=[WeathersCoarse.clear],
                                                timeofdays_coarse=[TimesOfDayCoarse.daytime],
                                                backend=ZipBackend(),
                                                classification_img_size=cfg['img_size']))
    print(f"{WeathersCoarse.clear.capitalize()} weather {TimesOfDayCoarse.daytime} time data val split size: {len(val_sets[-1])}")
    
    domains = []

    for timeofday in TIMEOFDAY_SEQUENCE:
        for weather in WEATHERS_SEQUENCE:
            if timeofday == TimesOfDayCoarse.daytime and weather == WeathersCoarse.clear:
                continue
            
            print(f"Loading {weather} weather {timeofday} time of day data...")

            train_sets.append(SHIFTClassificationDataset(split='train',
                                            data_root=cfg['data_root'],
                                            transforms=transforms_test,
                                            weathers_coarse=[weather],
                                            timeofdays_coarse=[timeofday],
                                            backend=ZipBackend(),
                                            classification_img_size=cfg['img_size']))
            print(f"{weather.capitalize()} weather {timeofday} time data train split size: {len(train_sets[-1])}")

            domains.append(f"{timeofday}_{weather}")

    return train_sets, val_sets, domains

def domain_to_experience_idx(domain):
    if domain in STANDARD_SHIFT_MIX_DOMAIN_SEQUENCE:
        return STANDARD_SHIFT_MIX_DOMAIN_SEQUENCE.index(domain)
    elif domain == 'daytime_clear':
        return len(STANDARD_SHIFT_MIX_DOMAIN_SEQUENCE)
