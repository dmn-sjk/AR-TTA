import os
import yaml
import numpy as np


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