"""
Code based on COTTA
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import PIL
import torchvision.transforms as transforms
import re
import numpy as np
import wandb

import utils.cotta_transforms as my_transforms
from utils.utils import split_up_model
from custom_bns.mecta_bn import MectaBN
from custom_bns.dynamic_bn import DynamicBN
from custom_bns.AdaMixBN import AdaMixBN


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class EmaTeacher(nn.Module):
    def __init__(self, model, optimizer, cfg: dict, steps=1, img_size: int = 64,
                 distillation_out_temp: int = 1, features_distillation_weight: float = 1):
        super().__init__()
        
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.features_distillation_weight = features_distillation_weight
        self.cfg = cfg
        self.num_replay_samples = self.cfg['batch_size']
        self.max_entropy_value = np.log(cfg['num_classes'])
        self.num_samples_update = 0
        self.current_model_probs = None
        assert steps > 0, "custom requires >= 1 step(s) to forward and update"
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_source = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
        # for mod_name, module in self.model.named_modules():
        #     if isinstance(module, MectaBN) or isinstance(module, DynamicBN):
        #         module.adapt_bn_stats = False
                # module.track_running_stats = False
                # module.running_mean = None
                # module.running_var = None
                
        self.adapt = True
        self.distillation_out_temp = distillation_out_temp
        
        self.step = 0
        
        # self.encoder, self.classifier = split_up_model(self.model, self.cfg['model'], encoder_out_relu_to_classifier=False)

    def forward(self, x):
        if self.adapt:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        else:
            outputs = self.model_ema(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
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

        ema_outputs = self.model_ema(x)
        student_outputs = self.model(x)
        # feats = self.encoder(x) 
        # student_outputs = self.classifier(feats) 
                
        pseudo_labels = ema_outputs.clone().detach()

        entropies = softmax_entropy(student_outputs / self.distillation_out_temp, pseudo_labels / self.distillation_out_temp, True)
        # entropies = softmax_entropy(student_outputs / self.distillation_out_temp, student_outputs / self.distillation_out_temp, True)
        # entropies = softmax_entropy(student_outputs / self.distillation_out_temp, labels / self.distillation_out_temp, False)
        
        loss = entropies.mean(0)

        loss.backward()
        
        
        # /* STATS DIST VISUALIZATION
        if 'wandb' in self.cfg.keys() and self.cfg['wandb']:
            i = 0
            # dist_source_test, dist_test_used, dist_source_used = 0, 0, 0
            for mod_name, module in self.model.named_modules():
                if isinstance(module, DynamicBN) or isinstance(module, MectaBN):
                    if isinstance(module.beta, torch.nn.Parameter):
                        beta = module.beta.item()
                        grad = module.beta.grad
                    else:
                        beta = module.beta
                        grad = None
                    
                    wandb.log({f"BN{i}_beta": beta}, step=self.step)
                    if grad is not None:
                        wandb.log({f"BN{i}_betaGradient": grad}, step=self.step)
                    # dist_source_test += gauss_symm_kl_divergence(module.test_mean.cuda(), module.test_var.cuda(),
                    #                     module.saved_running_mean.cuda(), module.saved_running_var.cuda(), 
                    #                     eps=1e-3).mean()
                    # dist_test_used += gauss_symm_kl_divergence(module.test_mean.cuda(), module.test_var.cuda(),
                    #                     module.running_mean.cuda(), module.running_var.cuda(), 
                    #                     eps=1e-3).mean()
                    # dist_source_used += gauss_symm_kl_divergence(module.saved_running_mean.cuda(), module.saved_running_var.cuda(),
                    #                     module.running_mean.cuda(), module.running_var.cuda(), 
                    #                     eps=1e-3).mean()
                    i += 1
            # mean
            # dist_source_test /= i
            # dist_test_used /= i
            # dist_source_used /= i
            
            # wandb.log({'MeanKLDivDist_BNstats_source_test': dist_source_test}, step=self.step)
            # wandb.log({'MeanKLDivDist_BNstats_test_used': dist_test_used}, step=self.step)
            # wandb.log({'MeanKLDivDist_BNstats_source_used': dist_source_used}, step=self.step)
            
            self.step += 1
        
        # STATS DIST VISUALIZATION */
        
        
        optimizer.step()
        optimizer.zero_grad()


        # with torch.no_grad():
        #     student_outputs = self.model(x)
        #     optimizer.zero_grad()

        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.cfg['mt'])
        
        # for m in self.model.modules():
        #     if isinstance(m, AdaMixBN):
        #         m.reset()
                
        # for ema_m, student_m in zip(self.model_ema.modules(), self.model.modules()):
        #     if isinstance(ema_m, nn.BatchNorm2d):
        #         student_m.running_mean.data = ema_m.running_mean.data.clone()
        #         student_m.running_var.data = ema_m.running_var.data.clone()
                # ema_m.running_mean.data = student_m.running_mean.data.clone()
                # ema_m.running_var.data = student_m.running_var.data.clone()
        
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


def configure_model(model, params_for_update: list = None, num_first_blocks_for_update: int = -1):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    # model.train()
    model.eval()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)

    if num_first_blocks_for_update == 0 and params_for_update is not None:
        return model

    # enable all trainable
    # first module is the whole network
    i=0
    for module_name, module in list(model.named_modules())[1:]:
        if num_first_blocks_for_update != -1 and \
            ('layer' in module_name or 'block' in module_name):
            starting_string = module_name.split('.')[0]
            block_nr = int(re.search(r'\d+$', starting_string).group())
            if block_nr > num_first_blocks_for_update:
                break
            
        # if i == 0 \
            # or 'block1' in module_name:
            # or 'block2' in module_name:
            # or 'block3' in module_name:
            # if isinstance(module, nn.BatchNorm2d) or isinstance(module, DynamicBN):
        # if 'fc' in module_name \
        #     or 'block3' in module_name:
        # if 'block2' in module_name:
            # or 'block1' in module_name:
            # if isinstance(module, nn.BatchNorm2d) or isinstance(module, DynamicBN):
            # module.requires_grad_(True)
        
        # if i < 4 \
        #     or 'layer1' in module_name:
        #     # or 'layer2' in module_name:
        #         if isinstance(module, nn.BatchNorm2d) or isinstance(module, DynamicBN):
        #             module.requires_grad_(True)
        
        # if 'fc' in module_name \
        #     or 'layer4' in module_name:
        #     # or 'layer3' in module_name:
        #         if isinstance(module, nn.BatchNorm2d) or isinstance(module, DynamicBN):
        #             module.requires_grad_(True)
        
        # print(i)
        # i+=1
        # print(module_name + '\n')

        if params_for_update is not None:
            for param_name, param in module.named_parameters():
                if f"{module_name}.{param_name}" in params_for_update:
                    param.requires_grad_(True)
        else:
            # if isinstance(module, DynamicBN) or isinstance(module, MectaBN):
            #     module.beta.requires_grad_(True)
            module.requires_grad_(True)
            
            # if isinstance(module, nn.BatchNorm2d) and not isinstance(module, MectaBN):
            # #     # force use of batch stats in train and eval modes
            #     print("BN stats reset")
            #     module.requires_grad_(True)
            #     module.track_running_stats = False
            #     module.running_mean = None
            #     module.running_var = None

    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"