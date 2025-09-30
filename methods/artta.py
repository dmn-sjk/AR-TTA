from methods.tta_method import TTAMethod
from utils.math import update_ema_variables
from datasets import get_source_dataset
from batch_norms.dynamic_bn import DynamicBN
from batch_norms import replace_bns

import torch
import numpy as np
from copy import deepcopy



def _softmax_entropy(x, x_ema, softmax_targets: bool = True):
    """Entropy of softmax distribution from logits."""
    if softmax_targets:
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
    else:
        return -(x_ema * x.log_softmax(1)).sum(1)

class ARTTA(TTAMethod, method_name='artta'):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
    
        self.model_ema = deepcopy(self.model)
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)
        for param in self.model_ema.parameters():
            param.detach_()
    
        self.alpha = self.cfg['alpha']
        self.memory = self.get_memory()
    
    @torch.enable_grad()
    def forward(self, x):
        with torch.no_grad():
            ema_outputs = self.model_ema(x)
                
        pseudo_labels = ema_outputs.detach().clone()

        x_for_model_update = x.clone()

        # whether to apply softmax on targets while calculating cross entropy
        softmax_targets = True

        # inject samples from memory with mixup
        if self.memory is not None:
            random_order_idxs = torch.randint(high=len(self.memory['labels']),
                                              size=(x_for_model_update.shape[0],))
            
            replay_x = self.memory['x'][random_order_idxs].to(self.cfg['device'])
            lam = np.random.beta(self.alpha, self.alpha)
            x_for_model_update = lam * x_for_model_update + (1 - lam) * replay_x

            # make accurate pseudo-labels for injected replay samples, since we have the labels
            replay_pseudo_labels = torch.nn.functional.one_hot(self.memory['labels'][random_order_idxs],
                                                               num_classes=self.cfg['num_classes'])\
                .to(torch.float32)\
                .to(self.cfg['device'])

            pseudo_labels = lam * pseudo_labels.softmax(1) + (1 - lam) * replay_pseudo_labels
            # here pseudo-labels are already after softmax
            softmax_targets = False
                
        student_update_out = self.model(x_for_model_update)

        entropies = _softmax_entropy(student_update_out, pseudo_labels, softmax_targets)
        loss = entropies.mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.cfg['mt'])

        return ema_outputs

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias', 'beta'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        self.model = replace_bns(self.cfg, self.model, DynamicBN)
        self.model = self.model.to(self.cfg['device'])
        self.model.eval()
        self.model.requires_grad_(True)
        
    def get_memory(self):
        memory = None
        if self.cfg['memory_size'] != 0:
            dataset = get_source_dataset(self.cfg)
            memory = {'x': torch.Tensor(), 'labels': torch.LongTensor()}

            # class-balanced memory
            for class_id in range(self.cfg['num_classes']):
                class_idxs = (torch.Tensor(dataset.targets) == class_id).nonzero(as_tuple=True)[0]
            
                memory_per_class = self.cfg['memory_size'] // self.cfg['num_classes']
                rest = self.cfg['memory_size'] % self.cfg['num_classes']
                if class_id < rest:
                    chosen_idxs = np.random.choice(class_idxs, memory_per_class + 1)
                else:
                    chosen_idxs = np.random.choice(class_idxs, memory_per_class)

                for idx in chosen_idxs:
                    # train_dataset[idx][0] - single sample image, adding dimension with None 
                    memory['x'] = torch.cat((memory['x'], dataset[idx][0][None,:]), dim=0)
                    memory['labels'] = torch.cat((memory['labels'], torch.LongTensor([dataset[idx][1]])), dim=0)
                    
            random_order_idxs = torch.randperm(len(memory['labels']))
            memory['x'] = memory['x'][random_order_idxs]
            memory['labels'] = memory['labels'][random_order_idxs]

        return memory

