import json
import os
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor

from datasets.domains import CORRUPTIONS_SEQ, SEVERITIES


class ImageNetCDataset(torch.utils.data.Dataset):
    NUM_CLASSES = 1000

    def __init__(self, data_root: str, corruption: str = None, 
                 split: str = "test", severity: int = 5, 
                 transform: Callable =  None, img_size: int = 224):
        self.transform = transform
        self.samples = []
        self.targets = []
        self.syn_to_class = {}
    
        self.resize_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
        ])
        
        if split == "train" and corruption is not None: 
            raise ValueError("Train split of CIFAR10 is not corrupted")


        with open(os.path.join(data_root, "ImageNet-C/imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        
        if corruption is None:
            if split == 'val':
                folders_dir = os.path.join(data_root, 'ImageNet/ILSVRC/Data/CLS-LOC/val_all_files')
            else:
                folders_dir = os.path.join(data_root, 'ImageNet/ILSVRC/Data/CLS-LOC', split)
        else:
            if severity not in SEVERITIES:
                raise ValueError("Severity level unavailable")

            if corruption not in CORRUPTIONS_SEQ:
                raise ValueError(f"Unknown corruption: {corruption}")
            
            folders_dir = os.path.join(data_root, "ImageNet-C", corruption, str(severity))

        if split == 'val' and corruption is None:
            with open(os.path.join(data_root, "ImageNet-C/ILSVRC2012_val_labels.json"), "rb") as f:
                val_to_syn = json.load(f)

            for entry in os.listdir(folders_dir):
                syn_id = val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(folders_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)      
        else:
            for syn_id in os.listdir(folders_dir):
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(folders_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        
        x = pil_to_tensor(x)
    
        x = x.to(torch.float32)
        x /= 255.0
        
        x = self.resize_image(x)

        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]