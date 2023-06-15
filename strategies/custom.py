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

import utils.cotta_transforms as my_transforms
from utils.intermediate_features import IntermediateFeaturesGetter


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


class Custom(nn.Module):
    def __init__(self, model, optimizer, cfg: dict, steps=1, img_size: int = 64,
                 distillation_out_temp: int = 1, features_distillation_weight: float = 1,
                 memory: dict = None, num_replay_samples: int = 10):
        super().__init__()
        
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.features_distillation_weight = features_distillation_weight
        self.memory = memory
        self.cfg = cfg
        self.num_replay_samples = num_replay_samples
        self.max_entropy_value = np.log(cfg['num_classes'])
        self.num_samples_update = 0
        self.current_model_probs = None
        assert steps > 0, "custom requires >= 1 step(s) to forward and update"
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_source = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
        if self.features_distillation_weight != 0:
            self.model = IntermediateFeaturesGetter(self.model)
            self.model_source = IntermediateFeaturesGetter(self.model_source)
            # name of penultimate layer
            self.features_layer = list(model.named_children())[-2][0]
            self.model.register_features(self.features_layer)
            self.model_source.register_features(self.features_layer)
            
        self.transform = get_tta_transforms(img_size=img_size)
        
        self.adapt = True
        self.distillation_out_temp = distillation_out_temp

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
        
        # source_outputs = self.model_source(x_for_source)
        ema_outputs = self.model_ema(x)
        
        # pseudo_labels = source_outputs.detach().clone()
        pseudo_labels = ema_outputs.detach().clone()

        if self.cfg['sampling_method'] is not None:
            if self.cfg['sampling_method'] in ['stochastic_entropy', 'stochastic_entropy_reverse']:
                entropies = softmax_entropy(pseudo_labels, pseudo_labels, softmax_targets=True)
                use_sample_probs = entropies / self.max_entropy_value

                if self.cfg['sampling_method'] == 'stochastic_entropy':
                    use_sample_probs = 1 - use_sample_probs
            
                chosen_samples_mask = torch.rand((x.shape[0],)) < use_sample_probs.cpu()

            elif self.cfg['sampling_method'] == 'random':
                chosen_samples_mask = torch.rand((x.shape[0],)) < 0.5

            elif self.cfg['sampling_method'] == 'eata':
                e_margin = 0.4 * np.log(self.cfg['num_classes'])
                d_margin = 0.4

                outputs = self.model(x)
                # adapt
                entropys = softmax_entropy(outputs, outputs, softmax_targets=True)
        
                chosen_samples_mask = torch.full((x.shape[0],), fill_value=False)
                
                # filter unreliable samples
                filter_ids_1 = torch.where(entropys < e_margin)
                chosen_samples_mask[filter_ids_1] = True
                ids1 = filter_ids_1
                entropys = entropys[filter_ids_1] 
                # filter redundant samples
                if self.current_model_probs is not None: 
                    cosine_similarities = torch.nn.functional.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
                    filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                    entropys = entropys[filter_ids_2]

                    temp_chosen_samples = torch.full((x.shape[0],), fill_value=False)
                    temp_chosen_samples[filter_ids_2] = True

                    chosen_samples_mask = torch.logical_and(chosen_samples_mask, temp_chosen_samples)
                    
                    updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
                    self.reset_model_probs(updated_probs)
                else:
                    updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
                    self.reset_model_probs(updated_probs)


            num_of_chosen_samples = chosen_samples_mask.int().sum().item()
            self.num_samples_update += num_of_chosen_samples
            if num_of_chosen_samples == 0:
                return self.model(x)
            
            x_for_model_update = x[chosen_samples_mask]
            x_for_source = x_for_model_update
            pseudo_labels = pseudo_labels[chosen_samples_mask]
        else:
            self.num_samples_update += x.shape[0]
        
        # inject samples from memory
        if self.memory is not None:
            random_order_idxs = torch.randint(high=len(self.memory['labels']),
                                              size=(self.num_replay_samples,))
            
            replay_x = self.memory['x'][random_order_idxs].to(self.cfg['device'])
            if self.cfg['replay_augs'] == 'mixup_from_memory':
                alpha = 0.4
                lam = np.random.beta(alpha, alpha)
                mixupped_x = lam * x_for_model_update + (1 - lam) * replay_x[:x_for_model_update.shape[0]]
                
                x_for_model_update = mixupped_x
            elif self.cfg['replay_augs'] == 'cotta':
                replay_x = self.transform(replay_x)

                x_for_source = torch.cat((x_for_source, replay_x), dim=0)
                x_for_model_update = x_for_source
                
            elif self.cfg['replay_augs'] == 'mixup_within_memory':
                alpha = 0.4
                lam = np.random.beta(alpha, alpha)
                random_idxs = torch.randperm(self.num_replay_samples)
                replay_x = lam * replay_x + (1 - lam) * replay_x[random_idxs]

                x_for_source = torch.cat((x_for_source, replay_x), dim=0)
                x_for_model_update = x_for_source
            else:
                raise ValueError(f"Unknown replay augs strategy name: {self.cfg['replay_augs']}")

                
        outputs_update = self.model(x_for_model_update)
        
        # # source_outputs = self.model_source(x_for_source)
        # ema_outputs = self.model_ema(x_for_source)
        
        # # pseudo_labels = source_outputs.detach().clone()
        # pseudo_labels = ema_outputs.detach().clone()
        
        # whether to apply softmax on targets while calculating cross entropy
        softmax_targets = True

        if self.features_distillation_weight != 0:
            source_features = self.model_source.get_features(self.features_layer)

        if self.memory is not None:
            # make accurate pseudo-labels for injected replay samples, since we have the labels
            replay_pseudo_labels = torch.nn.functional.one_hot(self.memory['labels'][random_order_idxs],
                                                               num_classes=self.cfg['num_classes'])\
                .to(torch.float32)\
                .to(self.cfg['device'])
                
            if self.cfg['replay_augs'] == 'cotta':
                # to have approximately one hot encoding after later softmax operation
                replay_pseudo_labels *= 1e6
                pseudo_labels[-self.num_replay_samples:] = replay_pseudo_labels
                softmax_targets = True

            elif self.cfg['replay_augs'] == 'mixup_within_memory':
                # to have approximately one hot encoding after later softmax operation
                replay_pseudo_labels = lam * replay_pseudo_labels + (1 - lam) * replay_pseudo_labels[random_idxs]
                pseudo_labels = pseudo_labels.softmax(1)
                pseudo_labels[-self.num_replay_samples:] = replay_pseudo_labels
                softmax_targets = False

            elif self.cfg['replay_augs'] == 'mixup_from_memory':
                pseudo_labels = lam * pseudo_labels.softmax(1) + (1 - lam) * replay_pseudo_labels[:pseudo_labels.shape[0]]
                softmax_targets = False
                
            else:
                raise ValueError(f"Unknown replay augs strategy name: {self.cfg['replay_augs']}")

        
        # anchor_prob = torch.nn.functional.softmax(self.model_source(x), dim=1).max(1)[0]
        # # Threshold choice discussed in supplementary
        # if anchor_prob.mean(0)<self.ap:
    # Augmentation-averaged Prediction
        # N = 32
        # outputs_emas = []
        # for i in range(N):
        #     outputs_  = self.model_source(self.transform(x)).detach()
        #     outputs_emas.append(outputs_)

        # augs_outputs = torch.stack(outputs_emas).mean(0)
        # else:
        #     outputs_ema = standard_ema

        loss = softmax_entropy(outputs_update / self.distillation_out_temp, pseudo_labels / self.distillation_out_temp, softmax_targets).mean(0)

        if self.features_distillation_weight != 0:
            loss += self.features_distillation_weight * nn.functional.mse_loss(torch.flatten(self.model.get_features(self.features_layer)),
                                                                               torch.flatten(source_features))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.cfg['mt'])
        # Stochastic restore

        if False:
            for nm, m  in self.model.named_modules():
                if 'model.' in nm:
                    nm = nm.replace('model.', '')
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)

        return self.model(x)

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


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
                if np in ['weight', 'bias'] and p.requires_grad:
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
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)

    if num_first_blocks_for_update == 0 and params_for_update is not None:
        return model
    
    # enable all trainable
    # first module is the whole network
    for module_name, module in list(model.named_modules())[1:]:
        if num_first_blocks_for_update != -1 and \
            ('layer' in module_name or 'block' in module_name):
            starting_string = module_name.split('.')[0]
            block_nr = int(re.search(r'\d+$', starting_string).group())
            if block_nr > num_first_blocks_for_update:
                break

        if params_for_update is not None:
            for param_name, param in module.named_parameters():
                if f"{module_name}.{param_name}" in params_for_update:
                    param.requires_grad_(True)
        else:
            module.requires_grad_(True)

        if isinstance(module, nn.BatchNorm2d):
            # force use of batch stats in train and eval modes
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

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