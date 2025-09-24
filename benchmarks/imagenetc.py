from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
)
from avalanche.benchmarks.scenarios import GenericCLScenario
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark

from constants.corrupted import CORRUPTIONS_SEQ
from . import register_benchmark


from robustbench.data import load_imagenetc
import torch



@register_benchmark("imagenetc")
def get_imagenetc_benchmark(cfg) -> GenericCLScenario:
    train_sets = []
    val_sets = []
    cfg['domains'] = CORRUPTIONS_SEQ

    for corruption in CORRUPTIONS_SEQ:
        x, y = load_imagenetc(
            severity=5,
            data_dir=cfg['data_root'],
            shuffle=False,
            corruptions=[corruption])
        imagenetc_dataset = torch.utils.data.TensorDataset(x,y)
        train_sets.append(imagenetc_dataset)

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
    if domain in CORRUPTIONS_SEQ:
        return CORRUPTIONS_SEQ.index(domain)
    elif domain == 'clear':
        return len(CORRUPTIONS_SEQ)
