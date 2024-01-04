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
import os
from torch.nn import functional as F

import utils.cotta_transforms as my_transforms
from utils.utils import split_up_model
from custom_bns.mecta_bn import MectaBN
from custom_bns.dynamic_bn import DynamicBN
from custom_bns.AdaMixBN import AdaMixBN
from utils.gmm import GaussianMixture
from utils.features_getter_wrapper import FeaturesGetterWrapper, features_return_nodes


class Nonparametric(nn.Module):
    def __init__(self, model, cfg: dict, steps=1):
        super().__init__()
        
        # FeaturesGetterWrapper.get_node_names(model)
        self.model = FeaturesGetterWrapper(model, features_return_nodes[cfg['model']])
        self.steps = steps
        self.cfg = cfg
        self.adapt = True
        self.step = 0
        
        self.dist_type = 'mahalonobis' # likelihood | mahalonobis | euclidan
        self.tukey = False
        self.shrinkage = True
        self.shrink1: float = 1.0
        self.shrink2: float = 1.0 # 1.0
        self.tukey1: float = 0.5
        self.covnorm = True
        self.norm_features = False
        prototypes_from = 'src'
        self.features_from = 'out_encoder'
        
        # online prototypes update
        if cfg['num_classes'] < 10:
            self.ema_length = 128 # 128
        else: 
            self.ema_length = 64 # 64
        self.buffer_size = 4096
        self.ema_n = torch.zeros(cfg['num_classes']).cuda()
        
        
        self.gmms_folder = 'experiment_data/gmms'
        n_features = torch.load(os.path.join(self.gmms_folder, cfg['dataset'], 'class' + str(0) + '.pth'))['mu'].shape[-1]
        self.gmms = []
        for _ in range(cfg['num_classes']):
            self.gmms.append(GaussianMixture(n_components=1, n_features=n_features,
                                             covariance_type='full', eps=1e-8).to(cfg['device']))
            
        self._load_gmms()
        for gmm in self.gmms:
            gmm.mu.data = F.normalize(gmm.mu, dim=-1)
            
            if self.shrinkage:
                gmm.var.data = self._shrink_cov(gmm.var.squeeze())[None, None,...]
            
            if self.covnorm:
                gmm.var.data = self._normalize_cov(gmm.var.squeeze())[None, None,...]
                
        
        prototypes_folder = 'experiment_data/prototypes'
        self.covs = torch.empty((0,), dtype=torch.float).to(cfg['device'])
        self.means = torch.empty((0,), dtype=torch.float).to(cfg['device'])
        for class_id in range(cfg['num_classes']):
            with open(os.path.join(prototypes_folder,  cfg['dataset'], prototypes_from, f'cov_class{class_id}.npy'), 'rb') as f: 
                cov = np.load(f)
                cov = torch.from_numpy(cov).to(cfg['device'])
                
                if self.shrinkage:
                    cov = self._shrink_cov(cov)
                    
                if self.covnorm:
                    cov = self._normalize_cov(cov)
                    
                self.covs = torch.cat([self.covs, cov[None, ...]], dim=0)
            with open(os.path.join(prototypes_folder,  cfg['dataset'], prototypes_from, f'mean_class{class_id}.npy'), 'rb') as f: 
                mean = np.load(f)
                mean = torch.from_numpy(mean).to(cfg['device'])
                if self.norm_features:
                    mean = F.normalize(mean, dim=-1)
                self.means = torch.cat([self.means, mean[None, ...]], dim=0)
                
    def forward(self, x):
        if self.adapt:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)
        else:
            outputs = self.model(x)

        return outputs
            
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        if isinstance(x, list):
            x, labels, _ = x
            
            labels = labels.to(self.cfg['device'])
            
            # labels = torch.nn.functional.one_hot(labels, num_classes=self.cfg['num_classes'])\
            #         .to(torch.float32)\
            #         .to(self.cfg['device'])
        
        with torch.no_grad(): 
            outputs = self.model(x)
            features = outputs[self.features_from]
            if features.dim() == 4:
                features = F.avg_pool2d(features, kernel_size=features.shape[-1]).squeeze()
            if self.norm_features:
                features = F.normalize(features, dim=-1)
            
        if self.tukey:
            features = self._tukey_transforms(features)
        
        if self.dist_type == 'likelihood':
            scores = self._get_scores(features)
        elif self.dist_type == 'mahalonobis':
            maha_dist = []
            for i in range(len(self.means)):
                dist = self._mahalanobis(features, self.means[i], self.covs[i])
                maha_dist.append(dist)

            scores = -torch.stack(maha_dist).T
        elif self.dist_type == 'euclidan':
            dists = []
            for i in range(len(self.means)):
                dist = torch.linalg.norm(features - self.means[i][None,...], dim=-1)
                dists.append(dist)
            scores = -torch.stack(dists).T
        else:
            raise NotImplementedError
        

        self._update_prototypes(features, labels)
        
        return scores

    def _get_scores(self, features: torch.Tensor):
        scores = torch.empty((0,)).to(self.cfg['device'])
        for gmm in self.gmms:
            score = gmm.score_samples(features)
            scores = torch.cat([scores, score[..., None]], dim=1)
        return scores
    
    def _get_predictions(self, features: torch.Tensor):
        scores = torch.empty((0,))
        for gmm in self.gmms:
            score = gmm.score_samples(features).cpu()
            scores = torch.cat([scores, score[..., None]], dim=1)
        return scores.argmax(dim=1)
    
    def _save_gmms(self):
        for i, gmm in enumerate(self.gmms):
            torch.save(gmm.state_dict(), os.path.join(self.gmms_folder, self.cfg['dataset'], 'class' + str(i) + '.pth'))
            
    def _load_gmms(self):
        for i, gmm in enumerate(self.gmms):
            # caviots with this gmm implementation
            gmm.params_fitted = True
            gmm.mu.data = gmm.mu.squeeze(0)
            gmm.load_state_dict(torch.load(os.path.join(self.gmms_folder, self.cfg['dataset'], 'class' + str(i) + '.pth')))

    def _tukey_transforms(self, x):
        if self.tukey1 == 0:
            return torch.log(x)
        else:
            return torch.pow(x, self.tukey1)
        
    def _mahalanobis(self, vectors, class_means, cov):
        x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(
            class_means, p=2, dim=-1
        )
        inv_covmat = torch.linalg.pinv(cov).float().to(vectors.device)
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0)
    
    def _normalize_cov(self, cov_mat):
        sd = torch.sqrt(torch.diagonal(cov_mat))  # standard deviations of the variables
        cov_mat = cov_mat / (torch.matmul(sd[..., None], sd[None, ...]))

        return cov_mat
    
    def _shrink_cov(self, cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0]).to(cov.device)
        cov_ = (
            cov
            + (self.shrink1 * diag_mean * iden)
            + (self.shrink2 * off_diag_mean * (1 - iden))
        )
        return cov_
    
    def _update_prototypes(self, features, labels):
        """
        https://github.com/Gorilla-Lab-SCUT/TTAC/blob/master/cifar/TTAC_onepass.py
        """
        
        b, d = features.shape
        feat_ext2_categories = torch.zeros(self.cfg['num_classes'], b, d).cuda() # K, N, D
        feat_ext2_categories.scatter_add_(dim=0, index=labels[None, :, None].expand(-1, -1, d), src=features[None, :, :])

        num_categories = torch.zeros(self.cfg['num_classes'], b, dtype=torch.int).cuda() # K, N
        num_categories.scatter_add_(dim=0, index=labels[None, :], src=torch.ones_like(labels[None, :], dtype=torch.int))

        self.ema_n += num_categories.sum(dim=1) # K
        alpha = torch.where(self.ema_n > self.ema_length, 
                            torch.ones(self.cfg['num_classes'], dtype=torch.float).cuda() / self.ema_length, 
                            1. / (self.ema_n + 1e-10))
        
        delta_pre = (feat_ext2_categories - self.means[:, None, :]) * num_categories[:, :, None] # K, N, D
        delta = alpha[:, None] * delta_pre.sum(dim=1) # K, D
        
        new_component_mean = self.means + delta
        new_component_cov = self.covs \
                            + alpha[:, None, None] * ((delta_pre.permute(0, 2, 1) @ delta_pre) - \
                                num_categories.sum(dim=1)[:, None, None] * self.covs) \
                            - delta[:, :, None] @ delta[:, None, :]
                            
        self.means = new_component_mean
        self.covs = new_component_cov