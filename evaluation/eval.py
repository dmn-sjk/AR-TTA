import os

import numpy as np
import torch
import yaml

from evaluation.evaluator import Evaluator
from utils.tensorboard_logger import TensorBoardLogger


def eval_domain(cfg, model, dataloader: torch.utils.data.DataLoader, logger: TensorBoardLogger):
    evaluator = Evaluator(cfg)
    
    for i, (x, y) in enumerate(dataloader):
        x = x.to(cfg['device'])
        
        preds = model(x)
        preds = preds.argmax(1).detach().cpu()

        acc, mca, acc_per_class = evaluator.add_preds(preds, y)

        log_dict = {f'acc_class_{i}': acc_per_class[i].item() for i in range(len(acc_per_class))}
        log_dict['acc'] = acc.item()
        log_dict['mca'] = mca.item()
        logger.log_scalars('per_batch', log_dict)
        
        if i == len(dataloader) - 1:
            overall_acc, overall_mca, overall_acc_per_class, num_samples, num_correct, \
                num_samples_per_class, num_correct_per_class = evaluator.get_summary()
            log_dict = {f'acc_class_{i}': overall_acc_per_class[i].item() for i in range(len(overall_acc_per_class))}
            log_dict['acc'] = overall_acc.item()
            log_dict['mca'] = overall_mca.item()
            logger.log_scalars('per_domain',log_dict)
        
        logger.step += 1
        
    return overall_acc, overall_mca, overall_acc_per_class, \
        num_samples, num_correct, num_samples_per_class, num_correct_per_class


def eval_seeds(log_dir: str):
    seed_results = {}
    
    for entity in os.listdir(log_dir):
        entity_path = os.path.join(log_dir, entity)
        if os.path.isdir(entity_path) and 'seed' in entity:
            overall_path = os.path.join(entity_path, 'overall.yaml')
            if os.path.exists(overall_path):
                with open(overall_path, 'r') as f:
                    seed_results[entity] = yaml.safe_load(f)
            else:
                print(f"No overall.yaml found for {entity}, omitting!")
    
    acc = np.mean([res['overall_acc'] for res in seed_results.values()]).item()
    amca = np.mean([res['AMCA'] for res in seed_results.values()]).item()
    return acc, amca, len(seed_results.keys())