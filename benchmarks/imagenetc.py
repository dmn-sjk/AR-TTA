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

from robustbench.data import load_imagenetc, load_imagenet
import torch
import os


def get_imagenetc_benchmark(cfg) -> GenericCLScenario:
    train_sets = []
    val_sets = []
    if cfg['benchmark'] in ["imagenetc_standard", "imagenetc_standard_subset"]:
        corruptions = copy(STANDARD_DOMAINS_SEQ)
    else:
        raise ValueError("Unknown type of cifar benchmark")

    cfg['domains'] = corruptions
    
    transforms_test = get_transforms(cfg, train=False)

    # val_sets.append(ImageNetCDataset(cfg['data_root'], corruption=None, split="val", transform=transforms_test,
    #                                  img_size=cfg['img_size']))
            
    for corruption in corruptions:
        if 'subset' in cfg['benchmark']:
            x, y = load_imagenetc(n_examples=5000, 
                                  severity=5, 
                                  data_dir=cfg['data_root'], 
                                  shuffle=False,
                                  corruptions=[corruption])
            
            imagenetc_dataset = torch.utils.data.TensorDataset(x,y)
            train_sets.append(imagenetc_dataset)
        else:
            train_sets.append(ImageNetCDataset(cfg['data_root'], corruption=corruption, split="val", transform=transforms_test,
                                        img_size=cfg['img_size']))      

    if cfg['end_with_source_domain']:
        if 'subset' in cfg['benchmark']:
            x, y = load_imagenet(n_examples=50, data_dir=os.path.join(cfg['data_root'], 'ImageNet/ILSVRC/Data/CLS-LOC'))
            imagenetc_dataset = torch.utils.data.TensorDataset(x,y)
            train_sets.append(imagenetc_dataset)
        else:
            train_sets.append(ImageNetCDataset(cfg['data_root'], corruption=None, split="val", transform=transforms_test,
                                        img_size=cfg['img_size']))
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
