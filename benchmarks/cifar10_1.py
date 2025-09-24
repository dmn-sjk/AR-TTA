from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
)
from avalanche.benchmarks.scenarios import GenericCLScenario
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark

from utils.transforms import get_transforms
from datasets.cifar10c import CIFAR10CDataset
from datasets.cifar10_1 import CIFAR101Dataset
from . import register_benchmark


@register_benchmark("cifar10_1")
def get_cifar10_1_benchmark(cfg) -> GenericCLScenario:
    train_sets = []
    val_sets = []

    cfg['domains'] = ['']
    
    transforms_test = get_transforms(cfg, train=False)
    
    val_sets.append(CIFAR10CDataset(cfg['data_root'], corruption=None, split="test", transforms=transforms_test))
    train_sets.append(CIFAR101Dataset(cfg['data_root'], transforms=transforms_test)) 

    train_exps_datasets = []
    for i, train_set in enumerate(train_sets):
        train_dataset_avl = make_classification_dataset(
            train_set,
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
            initial_transform_group="eval",
            task_labels=i
        )

        val_exps_datasets.append(
            classification_subset(val_dataset_avl)
        )

    return create_multi_dataset_generic_benchmark(train_datasets=train_exps_datasets,
                                                  test_datasets=val_exps_datasets,
                                                  )

def domain_to_experience_idx(domain):
    if domain == '':
        return 0
    elif domain == 'clear':
        return 1
