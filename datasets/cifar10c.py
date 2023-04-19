import torch
from typing import Callable
import os
import numpy as np

from constants.cifar import SEVERITIES, CORRUPTIONS



class CIFAR10CDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, corruption: str = None, 
                 split: str = "test", severity: int = 5, 
                 transforms: Callable =  None):
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

        self.images = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets)

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