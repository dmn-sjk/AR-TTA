from .cifar10c import CIFAR10CDataset
from .shift import SHIFTClassificationDataset
from .shift_dev.types import WeathersCoarse, TimesOfDayCoarse
import datasets.clad as clad
from .shift_dev.utils.backend import ZipBackend
from .cifar10_1 import CIFAR101Dataset
from .imagenetc import ImageNetCDataset
from robustbench.data import load_imagenetc
from utils.transforms import get_transforms

import torch
import numpy as np
import pickle


def get_test_dataloader(cfg, domain_dict: dict):
    transforms_test = get_transforms(cfg, train=False)

    if cfg['dataset'] in 'cifar10c':
        data = CIFAR10CDataset(cfg['data_root'], 
                               split="test", 
                               transforms=transforms_test, 
                               imbalanced=cfg['imbalanced'],
                               **domain_dict)        
        shuffle = True
    elif cfg['dataset'] == 'imagenetc':
        domain_dict['corruptions'] = [domain_dict['corruptions']]
        x, y = load_imagenetc(data_dir=cfg['data_root'],
                              shuffle=False,
                              **domain_dict)
        data = torch.utils.data.TensorDataset(x,y)
        shuffle = True
    elif cfg['dataset'] == 'shift':
        data = SHIFTClassificationDataset(split='train',
                                        data_root=cfg['data_root'],
                                        transforms=transforms_test,
                                        backend=ZipBackend(),
                                        classification_img_size=cfg['img_size'],
                                        **domain_dict)
        shuffle = False
    elif cfg['dataset'] == 'clad':
        data = clad.get_cladc_single_train_task(cfg["data_root"],
                                                transform=transforms_test, 
                                                img_size=cfg["img_size"],
                                                **domain_dict)
        shuffle = False
    elif cfg['dataset'] == 'cifar10_1':
        data = CIFAR101Dataset(cfg['data_root'], transforms=transforms_test)
        shuffle = True
    else:
        raise ValueError(cfg['dataset'])
    
    return torch.utils.data.DataLoader(data, 
                                       batch_size=cfg['batch_size'], 
                                       shuffle=shuffle,
                                       num_workers=cfg['num_workers'],
                                       pin_memory=True)
    
def get_source_dataset(cfg, train_split: bool = True):
    if cfg['dataset'] in ['cifar10c', 'cifar10_1']:
        dataset = CIFAR10CDataset(cfg['data_root'], 
                                        corruption=None, 
                                        split='train' if train_split else 'test', 
                                        transforms=None)
    elif cfg['dataset'] == 'shift':
        dataset = SHIFTClassificationDataset(split='train' if train_split else 'val',
                                                    data_root=cfg['data_root'],
                                                    transforms=None,
                                                    weathers_coarse=[WeathersCoarse.clear],
                                                    timeofdays_coarse=[TimesOfDayCoarse.daytime],
                                                    backend=ZipBackend(),
                                                    classification_img_size=cfg['img_size'])
    elif cfg['dataset'] == 'clad':
        dataset = clad.get_cladc_single_train_task(cfg['data_root'], 
                                                   task_id=0, 
                                                   transform=None)

        with open("datasets/clad_val_idxs.pkl", "rb") as f: 
            val_idxs = pickle.load(f)

        if train_split:
            train_mask = np.ones(len(dataset.ids), dtype=bool)
            train_mask[val_idxs] = False
            dataset.ids = list(np.array(dataset.ids)[train_mask])
            assert len(dataset.ids) == 4157
        else:
            dataset.ids = list(np.array(dataset.ids)[val_idxs])  
            assert len(dataset.ids) == 1000

    elif cfg['dataset'] == 'imagenetc':
        dataset = ImageNetCDataset(cfg['data_root'], 
                                         corruption=None, 
                                         split="train" if train_split else "val", 
                                         transform=None)
    else:
        raise NotImplementedError

    return dataset
