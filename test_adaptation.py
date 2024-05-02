import torch

from utils.utils import set_seed, get_experiment_name, get_experiment_folder, get_git_revision_hash, save_config
from utils.config_parser import ConfigParser
from benchmarks import get_benchmark
from strategies import get_strategy
from utils.evaluation import evaluate_results
import wandb


def main():
    cfg = ConfigParser(mode="tta").get_config()

    cfg['device'] = torch.device(f"cuda:{cfg['cuda']}"
                                 if torch.cuda.is_available() and cfg['cuda'] >= 0
                                 else "cpu")
    
    experiment_name = get_experiment_name(cfg)
    print(f"\n{(len(experiment_name) + 17) * '='}\nExperiment name: {experiment_name}\n{(len(experiment_name) + 17) * '='}\n")


    if cfg['dataset'] == 'cifar10c':
        shuffle = True
        from benchmarks.cifar10c import domain_to_experience_idx
    elif cfg['dataset'] == 'cifar10_1':
        shuffle = True
        from benchmarks.cifar10_1 import domain_to_experience_idx
    elif cfg['dataset'] == 'imagenetc':
        shuffle = True
        from benchmarks.imagenetc import domain_to_experience_idx
    elif cfg['dataset'] == 'clad':
        shuffle = False
        from benchmarks.cladc import domain_to_experience_idx
    elif cfg['dataset'] == 'shift':
        shuffle = False
        from benchmarks.shift import domain_to_experience_idx
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")


    seeds = [None]
    if 'seeds' in cfg.keys() and len(cfg['seeds']) > 0:
        seeds = cfg['seeds']

    for seed in seeds:
        if seed is not None:
            set_seed(seed)

        cfg['curr_seed'] = seed
        
        print(f"\n{17 * '='} Running seed: {seed} {17 * '='}\n")

        # reload on every seed in case of random thing happening inside
        benchmark = get_benchmark(cfg)
        strategy = get_strategy(cfg)

        if cfg['save_results']:
            # cfg['git_commit'] = get_git_revision_hash()
            save_config(cfg, experiment_name)

        if 'wandb' in cfg.keys() and cfg['wandb']:
            if wandb.run is not None:
                wandb.finish()
            
            job_type = f"{cfg['method']}_{cfg['model']}_{cfg['run_name']}"
            # Wandb's 64 letters limit
            if len(job_type) > 64:
                job_type = job_type[:64]
            wandb.init(project=cfg['project_name'],
                name='seed' + str(seed),
                job_type=job_type,
                group=cfg['dataset'] + '_TTA',
                config=cfg,
                resume="allow")


        for i, experience in enumerate(experience_generator(benchmark.train_stream, 
                                                        domains=cfg['domains'],
                                                        domains_to_exp_idx_func=domain_to_experience_idx)):
            # strategy.train(experience, eval_streams=[benchmark.streams['val']], shuffle=False,
            #                num_workers=cfg['num_workers)

            # if i == 0:
            #     print("Initial eval...")
            #     strategy.eval(benchmark.test_stream[0], num_workers=cfg['num_workers'])

            # avalanche cheat for correclty saving results
            experience.current_experience = 0
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
