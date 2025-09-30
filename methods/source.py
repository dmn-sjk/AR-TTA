import torch
import torch.nn as nn

from methods.tta_method import TTAMethod


class Source(TTAMethod, method_name='source'):
    def __init__(self, cfg, model: nn.Module):
        super().__init__(cfg, model)

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)
    
    def configure_model(self):
        self.model.eval()
    
    def collect_params(self):
        return [], []