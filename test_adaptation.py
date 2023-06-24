import torch

from utils.utils import set_seed, get_experiment_name, get_experiment_folder, get_git_revision_hash, save_config
from utils.config_parser import ConfigParser
from benchmarks import get_benchmark
from strategies import get_strategy
from utils.evaluation import evaluate_results


def main():
    cfg = ConfigParser(mode="tta").get_config()

    cfg['device'] = torch.device(f"cuda:{cfg['cuda']}"
                                 if torch.cuda.is_available() and cfg['cuda'] >= 0
                                 else "cpu")
    
    experiment_name = get_experiment_name(cfg)
    print(f"\n{(len(experiment_name) + 17) * '='}\nExperiment name: {experiment_name}\n{(len(experiment_name) + 17) * '='}\n")

    seeds = [None]
    if cfg['seeds'] is not None:
        seeds = cfg['seeds']
        
    config_saved = False

    for seed in seeds:
        if seed is not None:
            set_seed(seed)

        cfg['curr_seed'] = seed
        
        print(f"\n{17 * '='} Running seed: {seed} {17 * '='}\n")

        # reload on every seed in case of random thing happening inside
        benchmark = get_benchmark(cfg)
        strategy = get_strategy(cfg)

        if cfg['save_results'] and not config_saved:
            cfg['git_commit'] = get_git_revision_hash()
            save_config(cfg, experiment_name)

        if cfg['dataset'] == 'cifar10c':
            shuffle = True
            from benchmarks.cifar10c import domain_to_experience_idx
            
        elif cfg['dataset'] == 'clad':
            shuffle = False
            from benchmarks.cladc import domain_to_experience_idx

        elif cfg['dataset'] == 'shift':
            shuffle = False
            domain_to_experience_idx = None
        else:
            raise ValueError(f"Unknown dataset: {cfg['dataset']}")

        for i, experience in enumerate(experience_generator(benchmark.train_stream, 
                                                        domains=cfg['domains'],
                                                        domains_to_exp_idx_func=domain_to_experience_idx)):
            # strategy.train(experience, eval_streams=[benchmark.streams['val']], shuffle=False,
            #                num_workers=cfg['num_workers)

            # if i == 0:
            #     print("Initial eval...")
            #     strategy.eval(benchmark.test_stream[0], num_workers=cfg['num_workers'])

            strategy.train(experience, eval_streams=[], shuffle=shuffle,
                        num_workers=cfg['num_workers'])

            # print(f"Percentage of used samples: {(strategy.model.num_samples_update / len(experience.dataset)) * 100.0:.2f}")
            
            # strategy.model.num_samples_update = 0

            # strategy.eval(benchmark.test_stream[0], num_workers=cfg['num_workers'])
            
    evaluate_results(cfg)
        
def experience_generator(train_stream, domains, domains_to_exp_idx_func):
    if domains_to_exp_idx_func is not None:
        for domain in domains:
            idx = domains_to_exp_idx_func(domain)
            yield train_stream[idx]

    else:
        for exp in train_stream:
            yield exp

if __name__ == '__main__':
    main()
