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

def gauss_symm_kl_divergence(mean1, var1, mean2, var2, eps):
    # >>> out-place ops
    dif_mean = (mean1 - mean2) ** 2
    d1 = var1 + eps + dif_mean
    d1.div_(var2 + eps)
    d2 = (var2 + eps + dif_mean)
    d2.div_(var1 + eps)
    d1.add_(d2)
    d1.div_(2.).sub_(1.)
    # d1 = (var1 + eps + dif_mean) / (var2 + eps) + (var2 + eps + dif_mean) / (var1 + eps)
    return d1