import torch
from torch import nn

from avalanche.training.templates import SupervisedTemplate, BaseTemplate
import tent
from avalanche.core import SupervisedPlugin


# class TentPlugin(SupervisedPlugin):
#     def __init__(self, lr: float = 1e-3):
#         super().__init__()
#         self.lr = lr
#         self.model_tented = False
#         self.last_running_mean = []
#         self.last_running_var = []

#     def before_training(self, strategy: "SupervisedTemplate", **kwargs):
#         if not self.model_tented:
#             # self.last_running_mean =

#             model = tent.configure_model(strategy.model)
#             params, param_names = tent.collect_params(model)
#             optimizer = torch.optim.SGD(params, lr=self.lr)
#             tented_model = tent.Tent(model, optimizer)
#             strategy.model = tented_model
#             self.model_tented = True

#     def before_eval(self, strategy: "SupervisedTemplate", **kwargs):
#         if self.model_tented:

#             strategy.model = strategy.model.model
#             for param in strategy.model.parameters():
#                 param.requires_grad = False

#         self.model_tented = False


def get_tented_model_and_params(model):
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    params = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)

    return model, params


class TentPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.eval()
        for m in strategy.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
