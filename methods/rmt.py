
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import os
import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F 

from utils.utils import split_up_model
from datasets import get_source_dataset
from utils.cotta_transforms import get_tta_transforms
from utils.math import update_ema_variables
from .tta_method import TTAMethod


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) \
            - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

class RMT(TTAMethod, method_name='rmt'):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        
        batch_size_src = 64
        train_set = get_source_dataset(cfg)
        self.src_loader = DataLoader(train_set, 
                                     batch_size_src, 
                                     num_workers=cfg['num_workers'], 
                                     shuffle=True)
            

        self.num_classes = cfg['num_classes']
        self.device = cfg['device']
        self.src_loader_iter = iter(self.src_loader)
        self.contrast_mode = 'all'
        self.temperature = 0.1
        self.base_temperature = self.temperature
        self.projection_dim = 128
        self.lambda_ce_src = 1.0
        self.lambda_ce_trg = 1.0
        self.lambda_cont = 1.0
        self.m_teacher_momentum = 0.999
        # arguments neeeded for warm up
        self.warmup_steps = 50000
        self.final_lr = cfg['lr']
        arch_name = cfg['model']
        ckpt_path = "rmt_ckpts"
        self.dataset_name = cfg['dataset']

        self.tta_transform = get_tta_transforms(img_size=cfg['img_size'])  

        # setup loss functions
        self.symmetric_cross_entropy = SymmetricCrossEntropy()

        # Setup EMA model
        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)

        # define the prototype paths
        proto_dir_path = os.path.join(ckpt_path, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}_{self.dataset_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        # get source prototypes
        if os.path.exists(fname):
            print("Loading class-wise source prototypes...")
            self.prototypes_src = torch.load(fname)
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            features_src = torch.tensor([])
            labels_src = torch.tensor([])
            print("Extracting source prototypes...")
            with torch.no_grad():
                for data in tqdm.tqdm(self.src_loader):
                    x, y = data[0], data[1]
                    tmp_features = self.feature_extractor(x.to(self.device))
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src) > 100000:
                        break

            # create class-wise source prototypes
            self.prototypes_src = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src = torch.cat([self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

            torch.save(self.prototypes_src, fname)

        self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1)
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()

        # setup projector
        if self.dataset_name == "domainnet126":
            # do not use a projector since the network already clusters the features and reduces the dimensions
            self.projector = nn.Identity()
        else:
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
                                           nn.Linear(self.projection_dim, self.projection_dim)).to(self.device)
            self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})

        # warm up the mean-teacher framework
        if self.warmup_steps > 0:
            warmup_ckpt_path = os.path.join(ckpt_path, "warmup")
            ckpt_path = f"ckpt_warmup_{self.dataset_name}_{arch_name}_bs{batch_size_src}_lr{self.optimizer.param_groups[0]['lr']}.pth"
            ckpt_path = os.path.join(warmup_ckpt_path, ckpt_path)

            if os.path.exists(ckpt_path):
                print("Loading warmup checkpoint...")
                checkpoint = torch.load(ckpt_path)
                self.model.load_state_dict(checkpoint["model"])
                self.model_ema.load_state_dict(checkpoint["model_ema"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print(f"Loaded from {ckpt_path}")
            else:
                os.makedirs(warmup_ckpt_path, exist_ok=True)
                self.warmup()
                torch.save({"model": self.model.state_dict(),
                            "model_ema": self.model_ema.state_dict(),
                            "optimizer": self.optimizer.state_dict()
                            }, ckpt_path)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def warmup(self):
        print(f"Starting warm up...")
        for i in range(self.warmup_steps):
            #linearly increase the learning rate
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i+1) / self.warmup_steps

            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src, labels_src = batch[0], batch[1]
            imgs_src, labels_src = imgs_src.to(self.device), labels_src.to(self.device).long()

            # forward the test data and optimize the model
            outputs = self.model(imgs_src)
            outputs_ema = self.model_ema(imgs_src)
            loss = self.symmetric_cross_entropy(outputs, outputs_ema).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        print(f"Finished warm up...")
        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr   


    # Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    @torch.enable_grad()
    def forward(self, x):
        self.optimizer.zero_grad()
        
        # forward original test data
        features_test = self.feature_extractor(x)
        outputs_test = self.classifier(features_test)

        # forward augmented test data
        features_aug_test = self.feature_extractor(self.tta_transform(x))
        outputs_aug_test = self.classifier(features_aug_test)

        # forward original test data through the ema model
        outputs_ema = self.model_ema(x)

        with torch.no_grad():
            # dist[:, i] contains the distance from every source sample to one test sample
            dist = F.cosine_similarity(
                x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
                x2=features_test.view(1, features_test.shape[0], features_test.shape[1]).repeat(self.prototypes_src.shape[0], 1, 1),
                dim=-1)

            # for every test feature, get the nearest source prototype and derive the label
            _, indices = dist.topk(1, largest=True, dim=0)
            indices = indices.squeeze(0)

        features = torch.cat([
            self.prototypes_src[indices],
            features_test.view(features_test.shape[0], 1, features_test.shape[1]),
            features_aug_test.view(features_test.shape[0], 1, features_test.shape[1])], dim=1)
        loss_contrastive = self.contrastive_loss(features=features, labels=None)

        loss_self_training = (0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) + 0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)).mean(0)
        loss_trg = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive
        loss_trg.backward()

        if self.lambda_ce_src > 0:
            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            # train on labeled source data
            imgs_src, labels_src = batch[0], batch[1]
            features_src = self.feature_extractor(imgs_src.to(self.device))
            outputs_src = self.classifier(features_src)
            loss_ce_src = F.cross_entropy(outputs_src, labels_src.to(self.device).long())
            loss_ce_src *= self.lambda_ce_src
            loss_ce_src.backward()

        self.optimizer.step()

        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        # create and return the ensemble prediction
        return outputs_test + outputs_ema

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model"""
        # model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)