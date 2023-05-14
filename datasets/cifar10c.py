import torch
from typing import Callable
import os
import numpy as np

from constants.cifar import SEVERITIES, CORRUPTIONS



class CIFAR10CDataset(torch.utils.data.Dataset):
    NUM_CLASSES = 10

    def __init__(self, data_root: str, corruption: str = None, 
                 split: str = "test", severity: int = 5, 
                 transforms: Callable =  None, imbalanced: bool = False, imbalanced_rate: float = 0.01):
        self.dataset_path = os.path.join(data_root, "CIFAR-10-C")
        self.transforms = transforms
        self.imbalanced = imbalanced
        self.imbalanced_rate = imbalanced_rate

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
                raise ValueError(f"Unknown corruption: {corruption}")
            
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
        targets = np.load(os.path.join(path, self.label_filename))
        
        # change NHWC to NCHW format
        data = np.transpose(data, (0, 3, 1, 2))
        # make it compatible with our models (normalize)
        data = data.astype(np.float32) / 255.0

        if self.imbalanced:
            data, targets = self.produce_imbalanced_data(data, targets, self.imbalanced_rate)
            
        data = torch.from_numpy(data)
        # self. for avalanche 
        self.targets = torch.from_numpy(targets)

        self.dataset = torch.utils.data.TensorDataset(data,
                                                      self.targets)

    def produce_imbalanced_data(self, x, y, imbalanced_rate):
        rehearsal_data = None
        rehearsal_label = None

        data_percent = []
        data_num = int(x.shape[0] / self.NUM_CLASSES)

        for cls_idx in range(self.NUM_CLASSES):
            num = data_num * (imbalanced_rate ** (cls_idx / (self.NUM_CLASSES - 1)))
            data_percent.append(int(num))

        print("imbalance_ratio is {}".format(data_percent[0] / data_percent[-1]))
        print("per class num: {}".format(data_percent))

        self.class_list = data_percent

        for i in range(1, self.NUM_CLASSES + 1):
            a1 = y >= i - 1
            a2 = y < i
            index = a1 & a2
            task_train_x = x[index]
            label = y[index]
            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, data_percent[i - 1])
            tem_data = task_train_x[index]
            tem_label = label[index]
            if rehearsal_data is None:
                rehearsal_data = tem_data
                rehearsal_label = tem_label
            else:
                rehearsal_data = np.concatenate([rehearsal_data, tem_data], axis=0)
                rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

        shuffled_idxs = np.random.permutation(len(rehearsal_label))
        return rehearsal_data[shuffled_idxs], rehearsal_label[shuffled_idxs]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, target = self.dataset[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target.item()