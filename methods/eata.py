"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight. 
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_source_dataset
from methods.tta_method import TTAMethod
from utils.math import softmax_entropy


class EATA(TTAMethod, method_name='eata'):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

        self.fisher_alpha = self.cfg['fisher_alpha']
        self.e_margin = self.cfg['e_margin_coeff'] * math.log(cfg['num_classes'])
        self.d_margin = self.cfg['d_margin']
        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)
        self.calculate_fisher()

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return: 
        1. model outputs; 
        2. the number of reliable and non-redundant samples; 
        3. the number of reliable samples;
        4. the moving average  probability vector over all previous samples
        """
        # forward
        outputs = self.model(x)
        # adapt
        entropys = softmax_entropy(outputs, outputs)
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0]>-0.1)
        entropys = entropys[filter_ids_1] 
        # filter redundant samples
        if self.current_model_probs is not None: 
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = self.update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = self.update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # if x[ids1][ids2].size(0) != 0:
        #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1])**2).sum()
            loss += ewc_loss
        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        self.current_model_probs = updated_probs
        
        return outputs

    @staticmethod
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
        """Configure model for use with eata."""
        # train mode, because eata optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what eata updates
        self.model.requires_grad_(False)
        # configure norm for eata updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def calculate_fisher(self):
        fisher_batch_size = 64
        fisher_dataset = get_source_dataset(self.cfg)

        fisher_loader = torch.utils.data.DataLoader(fisher_dataset, 
                                                    batch_size=fisher_batch_size, 
                                                    shuffle=True, 
                                                    num_workers=self.cfg['num_workers'], 
                                                    pin_memory=True)

        ewc_optimizer = torch.optim.SGD(self.params, 0.001)
        self.fishers = {}
        train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
        stop = False
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):
            num_seen_samples_after_iter = iter_ * fisher_batch_size
            if num_seen_samples_after_iter > self.cfg['fisher_size']:
                num_samples_to_use = fisher_batch_size - (num_seen_samples_after_iter - self.cfg['fisher_size'])
                images = images[:num_samples_to_use]
                targets = targets[:num_samples_to_use]
                stop = True

            images = images.to(self.cfg['device'])
            targets = targets.to(self.cfg['device'])
            outputs = self.model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    self.fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()

            if stop:
                break

        print("Calculating fisher matrices finished")