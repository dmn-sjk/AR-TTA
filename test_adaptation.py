import torch
import os
import yaml

from utils.utils import get_experiment_folder, set_seed, get_experiment_name, save_config, get_seed_folder
from utils.config_parser import ConfigParser
from datasets import get_test_dataloader
from datasets.domains import get_domain_sequence
from evaluation.eval import eval_domain
from evaluation.evaluator import Evaluator
from evaluation.tensorboard_logger import TensorBoardLogger
from evaluation.eval_seeds import eval_seeds
from methods import get_method


def main():
    cfg = ConfigParser(mode="tta").get_config()

    cfg['device'] = torch.device(f"cuda:{cfg['cuda']}"
                                 if torch.cuda.is_available() and cfg['cuda'] >= 0
                                 else "cpu")
    
    experiment_name = get_experiment_name(cfg)
    print(f"\n{(len(experiment_name) + 17) * '='}\nExperiment name: {experiment_name}\n{(len(experiment_name) + 17) * '='}\n")

    seeds = [None]
    if 'seeds' in cfg.keys() and len(cfg['seeds']) > 0:
        seeds = cfg['seeds']

    for seed in seeds:
        if seed is not None:
            set_seed(seed)

        cfg['curr_seed'] = seed
        
        print(f"\n{17 * '='} Running seed: {seed} {17 * '='}\n")

        # benchmark = get_benchmark(cfg)
        domains = get_domain_sequence(cfg)
        cfg['domains'] = domains
        # strategy = get_strategy(cfg)
        model = get_method(cfg)

        if cfg['save_results']:
            save_config(cfg, experiment_name)

        log_dir = get_seed_folder(cfg)
        logger = TensorBoardLogger(os.path.join(log_dir, 'tb'))

        mcas = []
        evaluator = Evaluator(cfg)
        
        for i, domain in enumerate(domains):
            dataloader = get_test_dataloader(cfg, domain.get_domain_dict())
            acc, mca, acc_per_class, num_samples, num_correct, \
                num_samples_per_class, num_correct_per_class = eval_domain(cfg, model, dataloader, logger)
            print(f"\n{domain} accuracy: {acc:.1f}")
            print(f"{domain} MCA: {mca:.1f}")
            print(f"{domain} per-class accuracy: {[f'{acc:.1f}' for acc in acc_per_class.tolist()]}")
            
            evaluator.num_samples += num_samples
            evaluator.num_correct += num_correct
            evaluator.num_samples_per_class += num_samples_per_class
            evaluator.num_correct_per_class += num_correct_per_class
            mcas.append(mca)
        
        amca = torch.stack(mcas).nanmean().item()
        overall_acc, _, acc_per_class, _, _, _, _ = evaluator.get_summary()
        overall_acc = overall_acc.item()

        log_dict = {f'acc_class_{i}': acc_per_class[i].item() for i in range(cfg['num_classes'])}
        log_dict['acc'] = overall_acc
        log_dict['amca'] = amca
        logger.log_scalars('overall', log_dict)

        print(f"\nOverall accuracy: {overall_acc:.1f}")
        print(f"Overall per-class accuracy: {[f'{acc:.1f}' for acc in acc_per_class.tolist()]}")
        print(f"AMCA: {amca:.1f}")

        overall_results_path = os.path.join(log_dir, 'overall.yaml')
        overall_res_dict = {
            'overall_acc': overall_acc,
            'AMCA': amca
        }
        with open(overall_results_path, 'w') as f:
            yaml.safe_dump(overall_res_dict, f, default_flow_style=False)
    
    log_dir = get_experiment_folder(cfg)
    acc, amca, num_seeds = eval_seeds(log_dir)
    print('\n')
    print('='*30)
    print(f"Seed-averaged accuracy: {acc:.1f}")
    print(f"Seed-averaged AMCA: {amca:.1f}")
    print(f"Averaged over {num_seeds} {'seeds' if num_seeds > 1 else 'seed'}.")
    print('='*30)
        
    overall_results_path = os.path.join(log_dir, 'overall.yaml')
    overall_res_dict = {
        'overall_acc': acc,
        'AMCA': amca
    }
    with open(overall_results_path, 'w') as f:
        yaml.safe_dump(overall_res_dict, f, default_flow_style=False)
        
if __name__ == '__main__':
    main()
