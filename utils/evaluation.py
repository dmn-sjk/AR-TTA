import os
import json
import csv
import numpy as np
from prettytable import PrettyTable
from collections.abc import Iterable
from numbers import Number

from utils.utils import get_experiment_folder, get_experiment_name


def load_per_seed_results(experiment_name, experiment_folder, cfg):
    per_seed_paths = {}
    per_seed_results = {}
    for entity in os.listdir(experiment_folder):
        entity_path = os.path.join(experiment_folder, entity)
        if os.path.isdir(entity_path) and 'seed'in entity:
            results_path = os.path.join(entity_path, experiment_name + '_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    per_seed_results[entity] = json.load(f)
                    per_seed_paths[entity] = entity_path
            else:
                print(f"No results saved for {entity}, omitting!")
    
    # if there is no seed folders, check if there is results file in the main experiment dir
    if not len(per_seed_results.keys()):
        results_path = os.path.join(experiment_folder, experiment_name + '_results.json')
        if os.path.exists(results_path):
            print("No per seed results, using results with single seed from main experiment directory!")
            with open(results_path, 'r') as f:
                key = 'seed' + str(cfg['seed'])
                per_seed_results[key] = json.load(f)
                per_seed_paths[key] = experiment_folder
        else:
            raise RuntimeError(f"No results saved for experiment {experiment_name}!")
            
    return per_seed_results, per_seed_paths

def avg_per_seed_results(per_seed_results):
    results = {}
    
    def avg_per_seed(per_seed_results):
        if isinstance(per_seed_results[0], Number):
            return np.mean(per_seed_results)
        elif isinstance(per_seed_results[0], Iterable):
            # check if nested iterable
            if isinstance(per_seed_results[0][0], Iterable):
                averaged_out = []
                # recurence for each nested iterable
                for per_seed_iterables in zip(*[per_seed_results[i] for i in range(len(per_seed_results))]):
                    averaged_out.append(avg_per_seed(per_seed_iterables))
                return averaged_out
            else:
                return np.mean(per_seed_results, axis=0)
    
    seeds = list(per_seed_results.keys())
    
    for result_key in per_seed_results[seeds[0]].keys():
        results[result_key] = avg_per_seed([per_seed_results[seed][result_key] for seed in seeds])
    return results

def generate_evaluation(results, domains_sequence, domains_sequence_idxs, source_domain_idx, num_classes, save_folder, eval_name=''):
    table = [["class", *domains_sequence, "Avg. acc.", "Avg. mean class acc.", "Min class acc."], *[[] for i in range(num_classes + 1)]]
    
    if source_domain_idx is not None:
        table[0].extend(["Src. dom. avg acc.", "Src. dom. avg. mean class acc.", "Src. dom. min class acc."])


    # /* per class results */
    all_class_accs_matrix = np.empty(shape=(num_classes, len(domains_sequence)))
    source_domain_per_class_accs = []
    for class_id in range(num_classes):
        # class column
        table[class_id + 1].append(class_id)
        
        class_accs_key = "Top1_ClassAcc_Epoch/train_phase/train_stream/" + str(class_id)

        class_accs = np.array(results[class_accs_key])[domains_sequence_idxs] * 100.0
        all_class_accs_matrix[class_id] = class_accs

        # domains_sequence columns
        table[class_id + 1].extend(list(class_accs))
        # Avg acc column
        table[class_id + 1].append(np.nanmean(class_accs))
        # Avg mean class acc and Min class acc columns
        table[class_id + 1].extend(['-', '-'])

        if source_domain_idx is not None:
            source_domain_acc = results[class_accs_key][source_domain_idx] * 100.0
            # Source domain avg acc column
            table[class_id + 1].append(source_domain_acc)
            source_domain_per_class_accs.append(source_domain_acc)
            # Source domain avg mean class acc and ain class acc columns
            table[class_id + 1].extend(['-', '-'])

    # /* all classes results */
    # class column
    table[num_classes + 1].append('all')

    batchwise_acc_key = "Top1_Acc_MB/train_phase/train_stream"
    
    flattened_batchwise_accs = []
    for domain_idx in domains_sequence_idxs:
        domain_batchwise_accs = np.array(results[batchwise_acc_key][domain_idx]) * 100.0

        table[num_classes + 1].append(np.mean(domain_batchwise_accs))
        
        flattened_batchwise_accs.extend(domain_batchwise_accs)
        
    # Avg acc column
    table[num_classes + 1].append(np.mean(flattened_batchwise_accs))
    
    # Avg mean class acc and Min class acc columns
    table[num_classes + 1].append(all_class_accs_matrix.mean())
    table[num_classes + 1].append(all_class_accs_matrix.min())
 
    if source_domain_idx is not None:
        source_domain_acc = results[batchwise_acc_key][source_domain_idx]
        # Source domain avg acc column
        table[num_classes + 1].append(np.mean(source_domain_acc) * 100.0)
        # Source domain avg mean class acc and Min class acc columns
        table[num_classes + 1].append(np.mean(source_domain_per_class_accs))
        table[num_classes + 1].append(np.min(source_domain_per_class_accs))


    # save table
    with open(os.path.join(save_folder, 'evaluation.csv'), 'w') as f:
        writer = csv.writer(f)
        for row in table:
            writer.writerow(row)


    # print table to std out in compact version
    table_compact_version = []
    for row in table:
        table_compact_version.append([row[0], *row[16:]])
    tab = PrettyTable(table_compact_version[0], float_format='.2', title=eval_name + 'evaluation results')
    tab.add_rows(table_compact_version[1:])
    print(tab)


def evaluate_results(cfg):
    experiment_name = get_experiment_name(cfg)
    experiment_folder = get_experiment_folder(cfg)
    
    if cfg['end_with_source_domain']:
        domains_sequence = cfg['domains'][:-1]
        source_domain_idx = len(cfg['domains']) - 1
    else:
        domains_sequence = cfg['domains']
        source_domain_idx = None
        
    domains_sequence_idxs = list(range(len(domains_sequence)))

    per_seed_results, per_seed_paths = load_per_seed_results(experiment_name, experiment_folder, cfg)

    for (seed, seed_results), seed_path in zip(per_seed_results.items(), per_seed_paths.values()):
        generate_evaluation(seed_results, domains_sequence, domains_sequence_idxs, source_domain_idx, 
                            cfg['num_classes'], save_folder=seed_path, eval_name=seed + ' ')

    if len(per_seed_paths) > 1:
        results = avg_per_seed_results(per_seed_results)

        # random seed-averaged check
        for random_domain, random_batch in zip(np.random.randint(len(domains_sequence) - 1, size=(10,)),
                                            np.random.randint(500, size=(10,))):
            assert results["Top1_Acc_MB/train_phase/train_stream"][random_domain][random_batch] == \
                np.mean([result["Top1_Acc_MB/train_phase/train_stream"][random_domain][random_batch]
                        for result in per_seed_results.values()]), "Something is wrong with seed-averaging results"

        generate_evaluation(results, domains_sequence, domains_sequence_idxs, source_domain_idx, 
                            cfg['num_classes'], save_folder=experiment_folder, eval_name='seed-averaged ')
