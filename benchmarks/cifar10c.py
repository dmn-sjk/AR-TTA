from utils.transforms import get_transforms
import torch
import numpy as np
from constants.cifar import SEVERITIES, CORRUPTIONS, LONG_DOMAINS_SEQ, REPETITIVE_DOMAINS_SEQ
import os
from typing import Callable
from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
)
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark


class CIFAR10CDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, corruption: str = None, 
                 split: str = "test", severity: int = 5, transforms: Callable =  None):
        self.dataset_path = os.path.join(data_root, "CIFAR-10-C")
        self.transforms = transforms

        if split == "train": 
            if corruption is not None:
                raise ValueError("Train split of CIFAR10 is not corrupted")
            self.sub_path1 = 'origin'
            self.sub_path2 = ''
            self.data_filename = 'original.npy'
            self.label_filename = 'labels.npy'

        elif split == "test":
            if severity not in SEVERITIES:
                raise ValueError("Severity level unavailable")

            if corruption not in CORRUPTIONS and corruption is not None:
                raise ValueError("Unknown corruption")
            
            if corruption is None:
                # uncorrupted test set
                self.sub_path1 = 'corrupted'
                self.sub_path2 = 'severity-1'  # all data are same in 1~5
                self.data_filename = 'test.npy'
                self.label_filename = 'labels.npy'
            else:
                self.sub_path1 = 'corrupted'
                self.sub_path2 = 'severity-' + str(severity)
                self.data_filename = corruption + '.npy'
                self.label_filename = 'labels.npy'

        self.preprocessing()

    def preprocessing(self):

        path = os.path.join(self.dataset_path, self.sub_path1, self.sub_path2)

        data = np.load(os.path.join(path, self.data_filename))
        # change NHWC to NCHW format
        data = np.transpose(data, (0, 3, 1, 2))
        # make it compatible with our models (normalize)
        data = data.astype(np.float32) / 255.0

        self.images = torch.from_numpy(data)
        self.targets = torch.from_numpy(np.load(os.path.join(path, self.label_filename)))

        self.dataset = torch.utils.data.TensorDataset(self.images,
                                                      self.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, target = self.dataset[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target.item()
    
    # @property
    # def targets(self):
    #     """
    #     Get a list of all category ids, required for Avalanche.
    #     """
    #     return target


def get_cifar10c_benchmark(cfg):
    train_sets = []
    val_sets = []
    if cfg['benchmark'] == "cifar10c_long":
        corruptions = LONG_DOMAINS_SEQ
    elif cfg['benchmark'] == "cifar10c_repetitive":
        corruptions = REPETITIVE_DOMAINS_SEQ
    else:
        raise ValueError("Unknown type of cifar benchmark")
    
    transforms_test = get_transforms(cfg, train=False)
    
    val_sets.append(CIFAR10CDataset(cfg['data_root'], corruption=None, split="test", transforms=transforms_test))
    for corruption in corruptions:
        train_sets.append(CIFAR10CDataset(cfg['data_root'], corruption=corruption, split="test", transforms=transforms_test))        

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
