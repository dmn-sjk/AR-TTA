"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from utils.math import softmax_entropy
from .tta_method import TTAMethod
from utils.sam import SAM

from copy import deepcopy
import torch
import torch.nn as nn
import math
import numpy as np


class SAR(TTAMethod, method_name='sar'):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        
        self.optimizer = SAM(self.optimizer)

        self.margin_e0 = self.cfg['e_margin_coeff'] * math.log(self.cfg['num_classes'])
        self.reset_constant_em = 0.2  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        self.copy_model_and_optimizer()

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.ema = None

    def update_ema(self, new_data):
        if self.ema is None:
            self.ema = new_data
        else:
            with torch.no_grad():
                self.ema = 0.9 * self.ema + (1 - 0.9) * new_data

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward(self, x):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()
        # forward
        outputs = self.model(x)
        # adapt
        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = softmax_entropy(outputs, outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        self.optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = softmax_entropy(self.model(x), self.model(x))
        entropys2 = entropys2[filter_ids_1]  # second time forward  
        loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.update_ema(loss_second.item())  # record moving average loss values for model recovery

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)

        if self.ema is not None:
            if self.ema < self.reset_constant_em:
                print("ema < 0.2, now reset the model")
                self.reset()

        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def configure_model(self):
        """Configure model for use with SAR."""
        # train mode, because SAR optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what SAR updates
        self.model.requires_grad_(False)
        # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
