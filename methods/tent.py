from utils.math import softmax_entropy
from .tta_method import TTAMethod

import torch
import torch.nn as nn


class Tent(TTAMethod, method_name='tent'):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward(self, x):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        outputs = self.model(x)

        # adapt
        loss = softmax_entropy(outputs, outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        self.model.train()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
