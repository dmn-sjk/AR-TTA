from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
)
from avalanche.benchmarks.scenarios import GenericCLScenario
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from copy import copy
import numpy as np

from utils.transforms import get_transforms
from constants.corrupted import LONG_DOMAINS_SEQ, REPETITIVE_DOMAINS_SEQ, STANDARD_DOMAINS_SEQ
from datasets.imagenetc import ImageNetCDataset


def get_imagenetc_benchmark(cfg) -> GenericCLScenario:
    train_sets = []
    val_sets = []
    if cfg['benchmark'] in ["imagenetc_standard"]:
        corruptions = copy(STANDARD_DOMAINS_SEQ)
    else:
        raise ValueError("Unknown type of cifar benchmark")

    cfg['domains'] = corruptions
    
    transforms_test = get_transforms(cfg, train=False)

    val_sets.append(ImageNetCDataset(cfg['data_root'], corruption=None, split="val", transforms=transforms_test))
    for corruption in corruptions:
        train_sets.append(ImageNetCDataset(cfg['data_root'], corruption=corruption, split="val", transforms=transforms_test))        

    if cfg['end_with_source_domain']:
        train_sets.append(ImageNetCDataset(cfg['data_root'], corruption=None, split="val", transforms=transforms_test))
        cfg['domains'].append('clear')

    # transform_groups = dict(
    #     train=(transforms_train, None),
    #     eval=(transforms_test, None),
    # )

    train_exps_datasets = []
    for i, train_set in enumerate(train_sets):
        train_dataset_avl = make_classification_dataset(
            train_set,
            # transform_groups=transform_groups,
            initial_transform_group="train",
            task_labels=i
        )

        train_exps_datasets.append(
            classification_subset(train_dataset_avl)
        )

    val_exps_datasets = []
    for i, val_set in enumerate(val_sets):
        val_dataset_avl = make_classification_dataset(
            val_set,
            # transform_groups=transform_groups,
            initial_transform_group="eval",
            task_labels=i
        )

        val_exps_datasets.append(
            classification_subset(val_dataset_avl)
        )

    return create_multi_dataset_generic_benchmark(train_datasets=train_exps_datasets,
                                                  test_datasets=val_exps_datasets,
                                                  #   train_transform=transforms_train,
                                                  #   eval_transform=transforms_test
                                                  )

def domain_to_experience_idx(domain):
    if domain in STANDARD_DOMAINS_SEQ:
        return STANDARD_DOMAINS_SEQ.index(domain)
    elif domain == 'clear':
        return len(STANDARD_DOMAINS_SEQ)
