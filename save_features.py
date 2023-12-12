from utils.features_getter_wrapper import FeaturesGetterWrapper
from utils.intermediate_features import IntermediateFeaturesGetter
from benchmarks.cifar10c import CIFAR10CDataset
from benchmarks.imagenetc import ImageNetCDataset
from benchmarks.shift import SHIFTClassificationDataset
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from shift_dev.utils.backend import ZipBackend
from strategies import get_model
from utils.config_parser import ConfigParser

import os
import torch
from torch.nn import functional as F
import numpy as np


cfg = ConfigParser(mode="tta").get_config()

cfg['device'] = torch.device(f"cuda:{cfg['cuda']}"
                                if torch.cuda.is_available() and cfg['cuda'] >= 0
                                else "cpu")

model = get_model(cfg)

return_nodes = {
    'block1.layer.0.bn2': 'block1.layer.0.bn2',
    'block2.layer.2.bn2': 'block2.layer.2.bn2'
}

model = FeaturesGetterWrapper(model, return_nodes=return_nodes).cuda()

# model = IntermediateFeaturesGetter(model)
# for return_node in return_nodes:
#     model.register_features(return_node)


domain = 'gaussian_noise'
batch_size = 10
src_data = CIFAR10CDataset(cfg['data_root'], corruption=None, split='test', transforms=None)
corr_data = CIFAR10CDataset(cfg['data_root'], corruption=domain, split='test', transforms=None)

src_loader = torch.utils.data.DataLoader(src_data, batch_size=batch_size, shuffle=False, 
                                                num_workers=cfg['num_workers'], pin_memory=True)
corr_loader = torch.utils.data.DataLoader(corr_data, batch_size=batch_size, shuffle=False, 
                                                num_workers=cfg['num_workers'], pin_memory=True)

# custom_model.encoder = custom_model.encoder.cuda()
features = {key: torch.empty((0,)) for key in return_nodes.keys()}

for imgs, _ in corr_loader:
    imgs = imgs.cuda()
    
    # feat = custom_model.encoder(imgs).detach().cpu()
    feat = model(imgs)
    # feat = model._features
    for key in feat.keys():
        curr_feat = feat[key].detach().cpu()
        curr_feat = F.avg_pool2d(curr_feat, kernel_size=curr_feat.shape[-1]).squeeze()
        features[key] = torch.cat([features[key], curr_feat], dim=0)

for key in features.keys():
    out_dir = os.path.join('experiment_data','features','cifar10c',key)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{domain}_mectaBN_bs{batch_size}.npy'), 'wb') as f:
    # with open(os.path.join(out_dir, f'src_frozen.npy'), 'wb') as f:
        np.save(f, features[key].numpy())
        
print("features saved")