"""
adapted from https://github.com/cccntu/minLoRA
"""


import torch
from torch import nn
import math


import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


class LoRAParametrization(nn.Module):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype))
        self.forward_fn = self.lora_forward

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, X):
        return X + torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))).view(X.shape) * self.scaling

    def forward(self, X):
        return self.forward_fn(X)

    def disable_lora(self):
        self.forward_fn = lambda x: x

    def enable_lora(self):
        self.forward_fn = self.lora_forward

    @classmethod
    def from_linear(cls, layer, rank_mode='fixed', rank_param=4, lora_dropout_p=0.0, lora_alpha=1):
        device = layer.weight.device
        fan_out, fan_in = layer.weight.shape
        
        rank = cls.get_rank(layer, rank_mode, rank_param)
        
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        ).to(device)

    @classmethod
    def from_conv2d(cls, layer, rank_mode='fixed', rank_param=4, lora_dropout_p=0.0, lora_alpha=1):
        device = layer.weight.device
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        
        rank = cls.get_rank(layer, rank_mode, rank_param)
        
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        ).to(device)

    @classmethod
    def from_embedding(cls, layer, rank_mode='fixed', rank_param=4, lora_dropout_p=0.0, lora_alpha=1):
        device = layer.weight.device
        fan_in, fan_out = layer.weight.shape
        
        rank = cls.get_rank(layer, rank_mode, rank_param)
        
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        ).to(device)
        
    @staticmethod
    def get_rank(layer, mode, rank_param):
        if isinstance(layer, nn.Conv2d):
            out_ch, in_ch, kernel_size, _ = layer.weight.shape
        elif isinstance(layer, nn.Linear):
            out_ch, in_ch = layer.weight.shape
        else:
            raise NotImplementedError
    
        U, S, Vh = torch.linalg.svd(layer.weight.reshape(out_ch, -1))
        
        if mode=='fixed':
            lora_rank = rank_param
        elif mode=='threshold':
            assert rank_param>=0
            lora_rank = torch.sum(S>rank_param)
        elif mode=='ratio':
            assert 1>=rank_param>=0
            min_s = torch.max(S)*rank_param
            lora_rank = torch.sum(S>min_s)
        elif mode=='percentile':
            assert 1>=rank_param>=0
            s_cum = torch.cumsum(S, dim=0)
            min_cum_sum = rank_param * torch.sum(S)
            lora_rank = torch.sum(s_cum<min_cum_sum)
        else:
            raise NotImplementedError
        lora_rank = max(1, lora_rank)
        lora_rank = min(out_ch, in_ch, lora_rank)
        
        lora_rank = int(lora_rank)
        print(lora_rank)
        return lora_rank


default_lora_config = {  # specify which layers to add lora to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=4),
    },
}


def apply_lora(layer, register=True, merge=False, lora_config=default_lora_config):
    """add lora parametrization to a layer, designed to be used with model.apply"""
    if register:
        if type(layer) in lora_config:
            for attr_name, parametrization in lora_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


def add_lora(model, lora_config=default_lora_config):
    """add lora parametrization to all layers in a model. Calling it twice will add lora twice"""
    model.apply(partial(apply_lora, lora_config=lora_config))
    device = next(model.parameters()).device


def add_lora_by_name(model, target_module_names, lora_config=default_lora_config):
    """Add LoRA parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_lora(layer, lora_config=lora_config)


def merge_lora(model):
    """merge lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=True))


def remove_lora(model):
    """remove lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=False))
  

def get_lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if name_is_lora(k)}
  
def name_is_lora(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["lora_A", "lora_B"]
    )

def get_lora_params(model, print_shapes=False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_lora)

def get_params_by_name(model, print_shapes=False, name_filter=None):
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            if print_shapes:
                print(n, p.shape)
            yield p


def extract_conv(
    weight: torch.Tensor,
    mode = 'fixed',
    mode_param = 0,
    device = 'cpu',
) -> tuple[nn.Parameter, nn.Parameter]:
    out_ch, in_ch, kernel_size, _ = weight.shape
    
    U, S, Vh = torch.linalg.svd(weight.reshape(out_ch, -1).to(device))
    
    if mode=='fixed':
        lora_rank = mode_param
    elif mode=='threshold':
        assert mode_param>=0
        lora_rank = torch.sum(S>mode_param)
    elif mode=='ratio':
        assert 1>=mode_param>=0
        min_s = torch.max(S)*mode_param
        lora_rank = torch.sum(S>min_s)
    elif mode=='percentile':
        assert 1>=mode_param>=0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum<min_cum_sum)
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    extract_weight_A = Vh.reshape(lora_rank, in_ch, kernel_size, kernel_size).cpu()
    extract_weight_B = U.reshape(out_ch, lora_rank, 1, 1).cpu()
    del U, S, Vh, weight
    return extract_weight_A, extract_weight_B


if __name__ == "__main__":
    x = torch.rand((3,5))

    model = nn.Linear(5, 10)
    model.eval()
    
    print([param.data for param in model.parameters()])
    
    
    print("input:", x.shape)
    out = model(x.clone())
    print("default out:", out.shape)
    
    add_lora(model)

    out = model(x)
    print("lora out:", out.shape)