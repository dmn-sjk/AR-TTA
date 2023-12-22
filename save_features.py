from utils.features_getter_wrapper import FeaturesGetterWrapper
from utils.intermediate_features import IntermediateFeaturesGetter
from benchmarks.cifar10c import CIFAR10CDataset
from benchmarks.imagenetc import ImageNetCDataset
from benchmarks.shift import SHIFTClassificationDataset
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from shift_dev.utils.backend import ZipBackend
from strategies import get_model
from utils.config_parser import ConfigParser
from utils.utils import split_up_model

import os
import torch
from torch.nn import functional as F
import numpy as np
from torchvision.transforms import transforms
import clad
import pickle
from copy import deepcopy


cfg = ConfigParser(mode="tta").get_config()

cfg['device'] = torch.device(f"cuda:{cfg['cuda']}"
                                if torch.cuda.is_available() and cfg['cuda'] >= 0
                                else "cpu")

model = get_model(cfg)

# encoder, classifier = split_up_model(model, cfg['model'])

# model = IntermediateFeaturesGetter(model)
# for return_node in return_nodes:
#     model.register_features(return_node)


batch_size = 10
if cfg['dataset'] == 'cifar10c':
    
    # "gaussian_noise",
    # "shot_noise",
    # "impulse_noise",
    # "defocus_blur",
    # "glass_blur",
    # "motion_blur",
    # "zoom_blur",
    # "snow",
    # "frost",
    # "fog",
    # "brightness",
    # "contrast",
    # "elastic_transform",
    # "pixelate",
    # "jpeg_compression"
    
    domain = 'jpeg_compression'
    src_test_data = CIFAR10CDataset(cfg['data_root'], corruption=None, split='test', transforms=None)
    src_train_data = CIFAR10CDataset(cfg['data_root'], corruption=None, split='train', transforms=None)
    
    corr_data = CIFAR10CDataset(cfg['data_root'], corruption=domain, split='test', transforms=None)
    
    return_nodes = {
        'bn1': 'out_encoder',
        # 'block1.layer.0.bn2': 'block1.layer.0.bn2',
        # 'block2.layer.2.bn2': 'block2.layer.2.bn2'
    }
    model = FeaturesGetterWrapper(model, return_nodes=return_nodes).cuda()
    
elif cfg['dataset'] == 'clad':
    transforms = transforms.Compose([])
    train_sets = clad.get_cladc_train(cfg["data_root"], transforms, cfg["img_size"], avalanche=False, sequence_type="all", 
                                end_with_source_domain=False)
    src_data = train_sets[0]
    
    with open("datasets/clad_val_idxs.pkl", "rb") as f: 
        test_idxs = pickle.load(f)

    src_test_data = deepcopy(src_data)
    src_test_data.ids = list(np.array(src_test_data.ids)[test_idxs])
    
    train_mask = np.ones(len(src_data.ids), dtype=bool)
    train_mask[test_idxs] = False
    src_train_data = deepcopy(src_data)
    src_train_data.ids = list(np.array(src_data.ids)[train_mask])

    assert len(src_train_data.ids) == 4157
    assert len(src_test_data.ids) == 1000

    domain_idx = 1 # starting from 1
    corr_data = train_sets[domain_idx]
    domain = 'T' + str(domain_idx)
    
    return_nodes = {
        'layer4.2.add': 'out_encoder',
        # 'layer1.0.bn2': 'layer1.0.bn2',
        # 'layer2.2.bn2': 'layer2.2.bn2'
    }
    
    model = FeaturesGetterWrapper(model, return_nodes=return_nodes).cuda()

# src_loader = torch.utils.data.DataLoader(src_data, batch_size=batch_size, shuffle=False, 
#                                                 num_workers=cfg['num_workers'], pin_memory=True)

src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=batch_size, shuffle=False, 
                                                num_workers=cfg['num_workers'], pin_memory=True)
src_test_loader = torch.utils.data.DataLoader(src_test_data, batch_size=batch_size, shuffle=False, 
                                                num_workers=cfg['num_workers'], pin_memory=True)

corr_loader = torch.utils.data.DataLoader(corr_data, batch_size=batch_size, shuffle=False, 
                                                num_workers=cfg['num_workers'], pin_memory=True)

features = {key: torch.empty((0,)) for key in return_nodes.values()}

for imgs, _ in src_test_loader:
    imgs = imgs.cuda()
    
    # feat = encoder(imgs)
    # feat = {'out_encoder': feat}
    feat = model(imgs)
    # feat = model._features
    for key in feat.keys():
        curr_feat = feat[key].detach().cpu()
        curr_feat = F.avg_pool2d(curr_feat, kernel_size=curr_feat.shape[-1]).squeeze()
        features[key] = torch.cat([features[key], curr_feat], dim=0)

for key in features.keys():
    out_dir = os.path.join('experiment_data','features',cfg['dataset'],key)
    os.makedirs(out_dir, exist_ok=True)
    # with open(os.path.join(out_dir, f'{domain}_BN_bs{batch_size}.npy'), 'wb') as f:
    # with open(os.path.join(out_dir, f'{domain}_frozen.npy'), 'wb') as f:
    with open(os.path.join(out_dir, f'src_test_frozen.npy'), 'wb') as f:
        np.save(f, features[key].numpy())
    print(f"{key}: {features[key].numpy().sum()}")
        
print("features saved")