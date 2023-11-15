from avalanche.training.plugins import EvaluationPlugin
from strategies.adapt_turnoff_plugin import AdaptTurnoffPlugin
from typing import Sequence
import torch
import numpy as np

from strategies import custom
from strategies.frozen_strategy import FrozenModel
from benchmarks.cifar10c import CIFAR10CDataset
from benchmarks.imagenetc import ImageNetCDataset
from benchmarks.shift import SHIFTClassificationDataset
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from shift_dev.utils.backend import ZipBackend
import clad
from utils.dynamic_bn import DynamicBN, count_bn, replace_bn



def get_custom_strategy(cfg, model: torch.nn.Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    
    params_for_update = None
    if cfg['choose_params_with_fisher']:
        fisher_batch_size = 64
        if cfg['dataset'] == 'cifar10c':
            fisher_dataset = CIFAR10CDataset(cfg['data_root'], corruption=None, split='test', transforms=None)
        elif cfg['dataset'] == 'shift':
            fisher_dataset = SHIFTClassificationDataset(split='val',
                                                        data_root=cfg['data_root'],
                                                        transforms=None,
                                                        weathers_coarse=[WeathersCoarse.clear],
                                                        timeofdays_coarse=[
                                                            TimesOfDayCoarse.daytime],
                                                        backend=ZipBackend(),
                                                        classification_img_size=cfg['img_size'])
        elif cfg['dataset'] == 'clad':
            # TODO: for now val set has all the domains, maybe modify for only daytime and depending on the possibilities match the weather with train set 
            fisher_dataset = clad.get_cladc_val(cfg['data_root'], transform=None)
        else:
            raise NotImplementedError
        
        fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=fisher_batch_size, shuffle=True, 
                                                        num_workers=cfg['num_workers'], pin_memory=True)

        ewc_optimizer = torch.optim.SGD(model.parameters(), 0.001)
        fishers = {}
        train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
        stop = False
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):
            
            num_seen_samples_after_iter = iter_ * fisher_batch_size
            if num_seen_samples_after_iter > cfg['fisher_size']:
                num_samples_to_use = fisher_batch_size - (num_seen_samples_after_iter - cfg['fisher_size'])
                images = images[:num_samples_to_use]
                targets = targets[:num_samples_to_use]
                stop = True
                
            if cfg['cuda'] is not None:
                images = images.cuda(cfg['cuda'], non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(cfg['cuda'], non_blocking=True)
            outputs = model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: fisher})
            ewc_optimizer.zero_grad()

            if stop:
                break
            
        print("compute fisher matrices finished")
        del ewc_optimizer
        del fisher_loader
            
        for key in fishers.keys():
            fishers[key] = fishers[key].mean()
            
        all_fisher_vals = torch.Tensor(list(fishers.values()))
        fishers_median = torch.median(all_fisher_vals)
        smaller_than_median_mask = all_fisher_vals < fishers_median
        params_for_update = []
        for param_name, fisher_val_is_smaller_than_median in zip(fishers.keys(), smaller_than_median_mask):
            if fisher_val_is_smaller_than_median:
                params_for_update.append(param_name)

    # /* dynamic BN
    n_repalced = replace_bn(model, cfg['model'],
                   beta=cfg['init_beta'],
                   bn_dist_scale=cfg['bn_dist_scale'],
                   smoothing_beta=cfg['smoothing_beta'],
                   )
    n_bn = count_bn(model)
    assert n_repalced == n_bn, f"Replaced {n_repalced} BNs but actually have {n_bn}. Need to update `replace_bn`."

    m_cnt = 0
    for m in model.modules():
        if isinstance(m, DynamicBN):
            m_cnt += 1
    assert n_repalced == m_cnt, f"Replaced {n_repalced} BNs but actually inserted {m_cnt} AccumBN."
    # dynamic BN */

    model = custom.configure_model(model, 
                                   params_for_update=params_for_update, 
                                   num_first_blocks_for_update=cfg['num_first_blocks_for_update'])
    params, param_names = custom.collect_params(model)


    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,
                                    lr=cfg['lr'],
                                    betas=(cfg['beta'], 0.999),
                                    weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, nesterov=cfg['nesterov'])
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    memory = None
    if cfg['replay_augs'] is not None and cfg['replay_augs'] not in ['none', 'None', 'null']:
        if cfg['dataset'] == 'cifar10c':
            train_dataset = CIFAR10CDataset(cfg['data_root'], corruption=None, split='train', transforms=None)
        elif cfg['dataset'] == 'shift':
            train_dataset = SHIFTClassificationDataset(split='train',
                                                        data_root=cfg['data_root'],
                                                        transforms=None,
                                                        weathers_coarse=[WeathersCoarse.clear],
                                                        timeofdays_coarse=[
                                                            TimesOfDayCoarse.daytime],
                                                        backend=ZipBackend(),
                                                        classification_img_size=cfg['img_size'])
        elif cfg['dataset'] == 'clad':
            # TODO: for now val set has all the domains, maybe modify for only daytime and depending on the possibilities match the weather with train set 
            train_dataset = clad.get_cladc_train(cfg['data_root'], transform=None, sequence_type='source',
                                                 img_size=cfg['img_size'])[0]
        elif cfg['dataset'] == 'imagenetc':
            train_dataset = ImageNetCDataset(cfg['data_root'], corruption=None, split="train", transform=None,
                                     img_size=cfg['img_size'])
        else:
            raise NotImplementedError

        memory = {'x': torch.Tensor(), 'labels': torch.LongTensor()}

        # class-balanced memory
        for class_id in range(cfg['num_classes']):
            class_idxs = (torch.Tensor(train_dataset.targets) == class_id).nonzero(as_tuple=True)[0]
            
            if 'memory_size' in cfg.keys():
                cfg['memory_per_class'] = int(cfg['memory_size'] / cfg['num_classes'])

                rest = cfg['memory_size'] % cfg['num_classes']
                if class_id < rest:
                    chosen_idxs = np.random.choice(class_idxs, cfg['memory_size'] // cfg['num_classes'] + 1)
                else:
                    chosen_idxs = np.random.choice(class_idxs, cfg['memory_size'] // cfg['num_classes'])
                    
            else:
                chosen_idxs = np.random.choice(class_idxs, cfg['memory_per_class'])

            for idx in chosen_idxs:
                # train_dataset[idx][0] - single sample image, adding dimension with None 
                memory['x'] = torch.cat((memory['x'], train_dataset[idx][0][None,:]), dim=0)
                memory['labels'] = torch.cat((memory['labels'], torch.LongTensor([train_dataset[idx][1]])), dim=0)
                
        # random memory
        # chosen_idxs = np.random.randint(low=0, high=len(train_dataset.targets), size=(cfg['memory_size'],))
        # for idx in chosen_idxs:
        #     memory['x'] = torch.cat((memory['x'], train_dataset[idx][0][None,:]), dim=0)
        #     memory['labels'] = torch.cat((memory['labels'], torch.LongTensor([train_dataset[idx][1]])), dim=0)
            
        random_order_idxs = torch.randperm(len(memory['labels']))
        memory['x'] = memory['x'][random_order_idxs]
        memory['labels'] = memory['labels'][random_order_idxs]

    custom_model = custom.Custom(model, optimizer, cfg,
                       steps=cfg['steps'],
                       img_size=cfg['img_size'],
                       distillation_out_temp=cfg['distillation_out_temp'],
                       features_distillation_weight=cfg['features_distillation_weight'],
                       memory=memory,
                       num_replay_samples=cfg['num_replay_samples'],
                       alpha=cfg['alpha'],
                       beta=cfg['beta'])

    plugins.append(AdaptTurnoffPlugin())

    return FrozenModel(
        custom_model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
        device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)

