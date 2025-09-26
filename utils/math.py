import torch.nn as nn
import torch


def update_ema_variables(ema_model: nn.Module, model: nn.Module, alpha_teacher: float):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

@torch.jit.script
def softmax_entropy(x, x_ema):
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)