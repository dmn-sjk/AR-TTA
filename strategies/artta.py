from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import PIL
import torchvision.transforms as transforms
import re
import numpy as np
import wandb
import os
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from collections import deque

import utils.cotta_transforms as my_transforms
from utils.intermediate_features import IntermediateFeaturesGetter
from utils.utils import split_up_model
from torch.nn.modules.batchnorm import BatchNorm2d


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False, img_size: int=64):
    img_shape = (img_size, img_size, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class ARTTA(nn.Module):
    def __init__(self, model, optimizer, cfg: dict, steps=1, img_size: int = 64,
                 memory: dict = None,
                 alpha: float = 0.4):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.memory = memory
        self.cfg = cfg
        self.max_entropy_value = np.log(cfg['num_classes'])
        self.num_samples_update = 0
        self.current_model_probs = None
        assert steps > 0, "artta requires >= 1 step(s) to forward and update"
        
        self.alpha = alpha

        self.model_state, self.optimizer_state, self.model_ema, self.model_source = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
        self.encoder, self.classifier = split_up_model(self.model, self.cfg['model'])
        self.transform = get_tta_transforms(img_size=img_size)
        
        self.adapt = True
        
        self.step = 0
        self.ema = None
        
    def forward(self, x):
        if self.adapt:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        else:
            outputs = self.model_ema(x)

        return outputs

    def reset(self, only_student=False):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        if not only_student:
            # Use this line to also restore the teacher model                         
            self.model_state, self.optimizer_state, self.model_ema, self.model_source = \
                copy_model_and_optimizer(self.model, self.optimizer)


    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        if isinstance(x, list):
            x, labels, _ = x
            
            labels = torch.nn.functional.one_hot(labels, num_classes=self.cfg['num_classes'])\
                    .to(torch.float32)\
                    .to(self.cfg['device'])

        with torch.no_grad():
            ema_outputs = self.model_ema(x)
            source_outputs = self.model_source(x)
            # student_outputs = self.model(x)
            student_features = self.encoder(x)
            student_outputs = self.classifier(student_features)
                
        pseudo_labels = ema_outputs.detach().clone()

        self.num_samples_update += x.shape[0]
        x_for_model_update = x

        # inject samples from memory
        if self.memory is not None:
            random_order_idxs = torch.randint(high=len(self.memory['labels']),
                                              size=(x_for_model_update.shape[0],))
            
            replay_x = self.memory['x'][random_order_idxs].to(self.cfg['device'])
            lam = np.random.beta(self.alpha, self.alpha)
            mixupped_x = lam * x_for_model_update + (1 - lam) * replay_x

            x_for_model_update = mixupped_x

        # whether to apply softmax on targets while calculating cross entropy
        softmax_targets = True

        if self.memory is not None:
            # make accurate pseudo-labels for injected replay samples, since we have the labels
            replay_pseudo_labels = torch.nn.functional.one_hot(self.memory['labels'][random_order_idxs],
                                                               num_classes=self.cfg['num_classes'])\
                .to(torch.float32)\
                .to(self.cfg['device'])
                
            pseudo_labels = lam * pseudo_labels.softmax(1) + (1 - lam) * replay_pseudo_labels
            softmax_targets = False
                
        # student_update_out = self.model(x_for_model_update)
        student_features = self.encoder(x_for_model_update)
        student_update_out = self.classifier(student_features)

        entropies = softmax_entropy(student_update_out, pseudo_labels, softmax_targets)
        
        loss = entropies.mean(0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.cfg['mt'])

        self.step += 1
        return ema_outputs
        

@torch.jit.script
def softmax_entropy(x, x_ema, softmax_targets: bool = True):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    if softmax_targets:
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
    else:
        return -(x_ema * x.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.
    Walk the model's modules and collect all parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias', 'beta'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    model.eval()
    model.requires_grad_(True)

    return model
