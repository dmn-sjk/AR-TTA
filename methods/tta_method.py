from utils.optim import get_optimizer
from . import _METHODS

import torch.nn as nn


class TTAMethod(nn.Module):
    def __init__(self, cfg, model: nn.Module):
        super().__init__()
                
        self.cfg = cfg
        self.model = model
    
        self.configure_model()
        self.params, _ = self.collect_params()

        self.optimizer = get_optimizer(cfg, self.params) if len(self.params) > 0 else None
    
    def forward(self):
        raise NotImplementedError
    
    def configure_model(self):
        raise NotImplementedError
    
    def collect_params(self):
        raise NotImplementedError
    
    @classmethod
    def __init_subclass__(cls, method_name, **kwargs):
        super().__init_subclass__(**kwargs)
        _METHODS[method_name] = cls
