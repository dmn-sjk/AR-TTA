import os
import json
import yaml
import csv
import numpy as np
from prettytable import PrettyTable
from collections.abc import Iterable
from numbers import Number
import wandb

from utils.utils import get_experiment_folder, get_experiment_name


def load_per_seed_results(experiment_name, experiment_folder, cfg):
    per_seed_paths = {}
    per_seed_results = {}
    for entity in os.listdir(experiment_folder):
        entity_path = os.path.join(experiment_folder, entity)
        if os.path.isdir(entity_path) and 'seed' in entity:
            results_path = os.path.join(entity_path, experiment_name + '_results.json')
            if os.path.exists(results_path):
                per_seed_paths[entity] = entity_path
                with open(results_path, 'r') as f:
                    per_seed_results[entity] = json.load(f)
            else:
                print(f"No results saved for {entity}, omitting!")
    
    # if there is no seed folders, check if there is results file in the main experiment dir
    if not len(per_seed_results.keys()):
        results_path = os.path.join(experiment_folder, experiment_name + '_results.json')
        if os.path.exists(results_path):
            print("No per seed results, using results with single seed from main experiment directory!")
            key = 'seed' + str(cfg['curr_seed'])
            per_seed_paths[key] = experiment_folder
            with open(results_path, 'r') as f:
                per_seed_results[key] = json.load(f)
        else:
            raise RuntimeError(f"No results saved for experiment {experiment_name}!")
            
    return per_seed_results, per_seed_paths

def validate_domain_sequences(per_seed_domains):
    # assure equal number of domains
    for seed_idx in range(len(per_seed_domains.keys())):
        if seed_idx >= len(per_seed_domains.keys()) - 1:
            break
        assert len(per_seed_domains[list(per_seed_domains.keys())[seed_idx]]) == \
            len(per_seed_domains[list(per_seed_domains.keys())[seed_idx + 1]]), "Unequal number of domains between seeds"

def load_per_seed_domains(per_seed_paths):
    per_seed_domains = {}
    for seed, path in per_seed_paths.items():
        with open(os.path.join(path, "domains.yaml"), "r") as f:
            per_seed_domains[seed] = yaml.safe_load(f)

    validate_domain_sequences(per_seed_domains)

    return per_seed_domains

def unify_domains_order(per_seed_results, per_seed_domains_sequence, source_domain_idx):
    # unify to the order of first seed
    target_domains_sequence = per_seed_domains_sequence[list(per_seed_domains_sequence.keys())[0]]
    unified_per_seed_results = {}
    
    for seed in per_seed_results.keys():
        domain_idxs_correct_order = []
        unified_per_seed_results[seed] = {}
        # if there are more occurences of single domain
        domain_counter = {}
        for domain in np.unique(target_domains_sequence):
            domain_counter[domain] = 0
        
        for target_domain in target_domains_sequence:
            domain_idxs = [idx for idx, domain in enumerate(per_seed_domains_sequence[seed]) if domain == target_domain]
            domain_idxs_correct_order.append(domain_idxs[domain_counter[target_domain]])
            domain_counter[target_domain] += 1

        for result_key in per_seed_results[seed].keys():
            unified_per_seed_results[seed][result_key] = []
            for idx in  domain_idxs_correct_order:
                unified_per_seed_results[seed][result_key].append(per_seed_results[seed][result_key][idx])

            if source_domain_idx is not None:
                unified_per_seed_results[seed][result_key].append(per_seed_results[seed][result_key][source_domain_idx])

    return unified_per_seed_results, target_domains_sequence

def avg_per_seed_results(per_seed_results):
    results = {}
    
    def avg_per_seed(per_seed_results):
        if isinstance(per_seed_results[0], Number):
            return np.nanmean(np.array(per_seed_results, dtype=np.float64))
        elif isinstance(per_seed_results[0], Iterable):
            # check if nested iterable
            if isinstance(per_seed_results[0][0], Iterable):
                averaged_out = []
                # recurence for each nested iterable
                for per_seed_iterables in zip(*[per_seed_results[i] for i in range(len(per_seed_results))]):
                    averaged_out.append(avg_per_seed(per_seed_iterables))
                return averaged_out
            else:
                return np.nanmean(np.array(per_seed_results, dtype=np.float64), axis=0)
    
    seeds = list(per_seed_results.keys())
    
    for result_key in per_seed_results[seeds[0]].keys():
        results[result_key] = avg_per_seed([per_seed_results[seed][result_key] for seed in seeds])
    return results

def generate_evaluation(results, domains_sequence, domains_sequence_idxs, source_domain_idx, num_classes, save_folder, eval_name = '',
                        stds = None):
    table = [["class", *domains_sequence, "Avg. acc.", "Avg. mean class acc.", "Min mean class acc."], *[[] for i in range(num_classes + 1)]]
    
    if source_domain_idx is not None:
        table[0].extend(["Src. dom. avg acc.", "Src. dom. avg. mean class acc."])

    # /* per class results */
    all_class_accs_matrix = np.empty(shape=(num_classes, len(domains_sequence)))
    source_domain_per_class_accs = []
    for class_id in range(num_classes):
        # class column
        table[class_id + 1].append(class_id)
        
        class_accs_key = "Top1_ClassAcc_Epoch/train_phase/train_stream/" + str(class_id)
        class_accs = np.array(results[class_accs_key], dtype=np.float64)[domains_sequence_idxs] * 100.0
        all_class_accs_matrix[class_id] = class_accs

        # domains_sequence columns
        table[class_id + 1].extend(list(class_accs))
        # Avg acc column
        table[class_id + 1].append(np.nanmean(class_accs))
        # Avg mean class acc and Min  mean class acc columns
        table[class_id + 1].extend(['-', '-'])

        if source_domain_idx is not None:
            source_domain_acc = results[class_accs_key][source_domain_idx]
            if source_domain_acc is None:
                source_domain_acc = np.nan
            else:
                source_domain_acc *= 100.0

            # Source domain avg acc column
            table[class_id + 1].append(source_domain_acc)
            source_domain_per_class_accs.append(source_domain_acc)
            # Source domain avg mean class acc column
            table[class_id + 1].append('-')

    # /* all classes results */
    main_results_all = {}

    # class column
    table[num_classes + 1].append('all')

    batchwise_acc_key = "Top1_Acc_MB/train_phase/train_stream"
    
    flattened_batchwise_accs = []
    for domain_idx in domains_sequence_idxs:
        domain_batchwise_accs = np.array(results[batchwise_acc_key][domain_idx]) * 100.0

        table[num_classes + 1].append(np.mean(domain_batchwise_accs))
        
        flattened_batchwise_accs.extend(domain_batchwise_accs)
        
    # Avg acc column
    avg_acc = np.mean(flattened_batchwise_accs)
    table[num_classes + 1].append(avg_acc)
    main_results_all['avg_acc'] = avg_acc
    
    # Avg mean class acc and Min class acc columns
    # first avg of classes then domains like AMCA in CLAD
    avg_class_accs_matrix = np.nanmean(all_class_accs_matrix, axis=0)
    avg_mean_class_acc = np.nanmean(avg_class_accs_matrix)
    min_mean_class_acc = np.nanmin(avg_class_accs_matrix)
    table[num_classes + 1].append(avg_mean_class_acc)
    table[num_classes + 1].append(min_mean_class_acc)
    main_results_all['avg_mean_class_acc'] = avg_mean_class_acc
    main_results_all['min_mean_class_acc'] = min_mean_class_acc
 
    if source_domain_idx is not None:
        source_domain_acc = results[batchwise_acc_key][source_domain_idx]
        # Source domain avg acc column
        source_domain_avg_acc = np.nanmean(source_domain_acc) * 100.0
        table[num_classes + 1].append(source_domain_avg_acc)
        main_results_all['source_domain_avg_acc'] = source_domain_avg_acc
        # Source domain avg mean class acc column
        source_domain_avg_mean_class_acc = np.nanmean(source_domain_per_class_accs)
        table[num_classes + 1].append(source_domain_avg_mean_class_acc)
        main_results_all['source_domain_avg_mean_class_acc'] = source_domain_avg_mean_class_acc


    # save table
    with open(os.path.join(save_folder, 'evaluation.csv'), 'w') as f:
        writer = csv.writer(f)
        for row in table:
            writer.writerow(row)


    # print table to std out in compact version
    table_compact_version = []

    if stds is not None:
        last_row_additonal_strs = []
        for key in stds.keys():
            last_row_additonal_strs.append(f" ({stds[key]:.2f})")
        
    for i, row in enumerate(table):
        # last row
        if i == len(table) - 1 and stds is not None:
            for i in range(len(row) -  (1 + len(domains_sequence))):
                col_idx = i + 1 + len(domains_sequence)
                row[col_idx] = f"{row[col_idx]:.2f}{last_row_additonal_strs[i]}"

        table_compact_version.append([row[0], *row[1 + len(domains_sequence):]])
        
    tab = PrettyTable(table_compact_version[0], float_format='.2', title=eval_name + 'evaluation results')
    tab.add_rows(table_compact_version[1:])
    print(tab)
    # save table
    with open(os.path.join(save_folder, 'evaluation.txt'), 'w') as f:
        f.write(str(tab))
    
    return main_results_all

def evaluate_results(cfg):
    experiment_name = get_experiment_name(cfg)
    experiment_folder = get_experiment_folder(cfg)

    per_seed_results, per_seed_paths = load_per_seed_results(experiment_name, experiment_folder, cfg)
    per_seed_domains = load_per_seed_domains(per_seed_paths)
    
    per_seed_domains_sequence = {}
    if cfg['end_with_source_domain']:
        for seed in per_seed_domains.keys():
            per_seed_domains_sequence[seed] = per_seed_domains[seed][:-1]
        # every seed always the same number of domains, the order might change
        source_domain_idx =  len(per_seed_domains[seed]) - 1
    else:
        for seed in per_seed_domains.keys():
            per_seed_domains_sequence[seed] = per_seed_domains[seed]
        source_domain_idx = None

    domains_sequence_idxs = list(range(len(per_seed_domains_sequence[list(per_seed_domains_sequence.keys())[0]])))

    main_results_all = {'avg_acc': [], 'avg_mean_class_acc': [], 'min_mean_class_acc': [], 
                        'source_domain_avg_acc': [], 'source_domain_avg_mean_class_acc': []}
    for (seed, seed_results), seed_path in zip(per_seed_results.items(), per_seed_paths.values()):
        main_results_all_seed = generate_evaluation(seed_results,
                                                    per_seed_domains_sequence[seed],
                                                    domains_sequence_idxs,
                                                    source_domain_idx,
                                                    cfg['num_classes'], save_folder=seed_path, eval_name=seed + ' ')
        
        if 'seed' + str(cfg['curr_seed']) == seed and cfg['wandb']:
            wandb.run.summary["avg_acc"] = main_results_all_seed['avg_acc']
            wandb.run.summary["avg_mean_class_acc"] = main_results_all_seed['avg_mean_class_acc']
        
        for key in main_results_all_seed.keys():
            main_results_all[key].append(main_results_all_seed[key])

    if len(per_seed_paths) > 1:
        if 'long_random' in cfg['benchmark']:
            # TODO:
            raise NotImplementedError("For now it is assumed that each seed has the same number of occurances. \
                Long random benchmarks break this assumption")

        # change the order of results to match domains in each seed, in case of different order of domains between seeds
        # (to meet the assumption in avg_per_seed_results func)
        if 'random' in cfg['benchmark']:
            per_seed_results, domains_sequence = unify_domains_order(per_seed_results, per_seed_domains_sequence, source_domain_idx)
        else:
            domains_sequence = per_seed_domains_sequence[list(per_seed_domains_sequence.keys())[0]]
        results = avg_per_seed_results(per_seed_results)

        # random seed-averaged check
        num_of_checks = 10

        if len(domains_sequence_idxs) > 1:
            random_domains = np.random.randint(len(domains_sequence_idxs) - 1, size=(num_of_checks,))
        else: 
            random_domains = [0] * num_of_checks

        for random_domain in random_domains:
            random_batch = np.random.randint(len(results["Top1_Acc_MB/train_phase/train_stream"][random_domain]))

            assert results["Top1_Acc_MB/train_phase/train_stream"][random_domain][random_batch] == \
                np.nanmean([result["Top1_Acc_MB/train_phase/train_stream"][random_domain][random_batch]
                        for result in per_seed_results.values()]), "Something is wrong with seed-averaging results"

        stds = {}
        for key in main_results_all.keys():
            stds[key] = np.std(main_results_all[key])

        generate_evaluation(results, domains_sequence, domains_sequence_idxs, source_domain_idx, 
                            cfg['num_classes'], save_folder=experiment_folder, eval_name='seed-averaged ',
                            stds=stds)
