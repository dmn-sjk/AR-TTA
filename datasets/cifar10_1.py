import torch
from typing import Callable
import os
import numpy as np
import pathlib


def load_cifar10_1_dataset(data_dir, version_string='v6'):
    """
    Integrated from: https://github.com/modestyachts/CIFAR-10.1/tree/master
    """

    filename = 'cifar10.1'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_dir, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_dir, imagedata_filename))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    return imagedata, labels


class CIFAR101Dataset(torch.utils.data.Dataset):
    NUM_CLASSES = 10

    def __init__(self, data_root: str, transforms: Callable =  None):
        self.dataset_path = os.path.join(data_root, "cifar10_1")
        self.transforms = transforms
        self.imgs, self.targets = load_cifar10_1_dataset(self.dataset_path)
        self._preprocess()

    def _preprocess(self):
        # change NHWC to NCHW format
        
        self.imgs = np.transpose(self.imgs, (0, 3, 1, 2))
        
        # make it compatible with our models (normalize)
        self.imgs = self.imgs.astype(np.float32) / 255.0

        self.imgs = torch.from_numpy(self.imgs)
        # self. for avalanche 
        self.targets = torch.from_numpy(self.targets)

        self.dataset = torch.utils.data.TensorDataset(self.imgs,
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