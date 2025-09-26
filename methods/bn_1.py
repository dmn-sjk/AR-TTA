from .tta_method import TTAMethod

import torch
import torch.nn as nn

class BN_1(TTAMethod, method_name='bn_1'):
    def __init__(self, cfg, model: nn.Module):
        super().__init__(cfg, model)

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)
    
    def configure_model(self):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    
    def collect_params(self):
        return [], []