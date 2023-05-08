import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import deque
from prettytable import PrettyTable
import argparse
import sys
import csv
import shutil
import yaml
import matplotlib.colors as mcolors


LOGS_TO_USE = []
RESULTS_FOLDER = 'results'
LOGS_FOLDER = 'logs'
DOMAINS_FILE = 'domains.txt'
WINDOW_SIZE = 500 # for batch-wise train acc plot  


def get_plot_color(method):
    if 'frozen' in method:
        return 'tab:red'
    elif 'finetune' in method:
        return 'tab:green'
    elif 'eata' in method:
        return 'tab:orange' 
    elif 'cotta' in method:
        return 'tab:blue' 
    elif 'tent' in method:
        return 'tab:purple' 

    return np.random.choice(list(mcolors.TABLEAU_COLORS)[5:])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_name', type=str, default=None, required=True,
                        help='')
    parser.add_argument('log_names', type=str, nargs='*',
                        help='name of the folder of log of the experiment')
    parser.add_argument('--save_results', action='store_true',
                        help='')
    args = parser.parse_args()
    
    for log in args.log_names:
        LOGS_TO_USE.append(log)
        
    if len(LOGS_TO_USE) == 0:
        raise ValueError("Provide log folders' names to generate results from") 
        
    return args

def get_and_check_domains():
    domains = []
    for log in LOGS_TO_USE:
        with open(os.path.join(LOGS_FOLDER, log, log + '_config.yaml'), "r") as f:
            curr_domains = yaml.safe_load(f)['domains']
            
            if len(curr_domains) == 0:
                raise ValueError("Empty domains file!")
            
            if len(domains) == 0:
                domains = curr_domains
                
            if domains != curr_domains:
                raise ValueError(f"Every used exeperiment should have the same domains in the same order:\n \
                                 - domains from previous logs: {domains}\n \
                                 - domains in {log} logs folder: {curr_domains}")
    
    return domains

def load_results():
    results = {}

    for log in LOGS_TO_USE:
        with open(os.path.join(LOGS_FOLDER, log, log + '_results.json'), 'r') as f:
            results[log] = json.load(f)
    
    return results

def copy_config_files():
    for log in LOGS_TO_USE:
        shutil.copy(os.path.join(LOGS_FOLDER, log, log + '_config.yaml'),
                    os.path.join(RESULTS_FOLDER, args.results_name))
        
def main(args):
    if args.save_results:
        os.makedirs(os.path.join(RESULTS_FOLDER, args.results_name), exist_ok=True)
        
    copy_config_files()
    domains = get_and_check_domains()
    results = load_results()

    # -----
    table = [["method", *domains, "Avg"], *[[] for i in range(len(LOGS_TO_USE))]]
    avg_accs_train = []
    print("Average acc train:")
    for i, method in enumerate(LOGS_TO_USE):
        table[i + 1].append(method)
        flattened_results = []
        for single_domain in results[method]["Top1_Acc_MB/train_phase/train_stream"]:
            table[i + 1].append(np.mean(single_domain) * 100.0)
            flattened_results.extend(single_domain)

        avg_acc_train = np.mean(flattened_results) * 100.0

        print(f'- {method}: {avg_acc_train:.2f}')
        table[i + 1].append(avg_acc_train)
        
        avg_accs_train.append(avg_acc_train)

    tab = PrettyTable(table[0])
    tab.add_rows(table[1:])
    tab.float_format = '.2'

    if args.save_results:
        with open(os.path.join(RESULTS_FOLDER, args.results_name, args.results_name + '.csv'), 'w') as f:
            writer = csv.writer(f)
            for row in table:
                writer.writerow(row)
    else:
        print(tab)

    # -----
    fig, ax = plt.subplots()
    avg_accs = []
    avg_times = []
    for i, method in enumerate(LOGS_TO_USE):
        avg_accs.append(avg_accs_train[i])
        avg_times.append(np.mean(results[method]["AvgTimeIter/train_phase/train_stream"]))

    width = 0.3
    plt.xticks(range(len(avg_accs)), LOGS_TO_USE, rotation=45)
    plt.title(f"Avg train accuracy and run time")
    ax.bar(np.array(range(len(avg_accs))) - (width / 2), avg_accs, color='blue', width=width)
    ax2 = ax.twinx()
    ax2.bar(np.array(range(len(avg_accs))) + (width / 2), avg_times, color='red', width=width)
    ax2.set_ylabel('Avg run time', color='red')
    plt.xlabel("After task")
    ax.set_ylabel("Accuracy [%]", color='blue')
    plt.tight_layout()

    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'avg_train_acc_and_time.png'))
    else:
        plt.show()

    # -----
    window = deque(maxlen=WINDOW_SIZE)

    # TRAINING SEQUENCES
    fig, ax = plt.subplots(figsize=(15, 10))
    for method in LOGS_TO_USE:
        whole_results = []
        window.clear()
        for task_results in results[method]["Top1_Acc_MB/train_phase/train_stream"]:
            window_accs = []
            window.clear()
            for batch_acc in task_results:
                window.append(batch_acc)
                window_accs.append(np.mean(window))
            whole_results.extend(window_accs)
        accs = np.array(whole_results) * 100.0
        ax.plot(range(len(whole_results)), accs, label=method, color=get_plot_color(method))

    end_of_x_axis = 0
    xticks = [end_of_x_axis]
    for task_results in results[LOGS_TO_USE[0]]["Top1_Acc_MB/train_phase/train_stream"]:
        task_samples = len(task_results)
        end_of_x_axis += task_samples
        xticks.append(end_of_x_axis)

    plt.xticks(xticks, [*domains, ''], rotation=45)
    plt.grid(axis='both')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title("Train sequences accuracy")
    plt.xlabel("Task")
    plt.ylabel("Accuracy [%]")
    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'train_acc_plot'))
    else:
        plt.show()

    # TEST AND VAL
    # checking only test stream
    seqs = ["Top1_Acc_Stream/eval_phase/test_stream"]
    for seq in seqs:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for method in LOGS_TO_USE:
            accs = np.array(results[method][seq]) * 100.0
            ax.plot(range(len(accs)), accs, marker='.', label=method, color=get_plot_color(method))

        plt.xticks(range(len(domains) + 1), ['Init', *range(len(domains))])
        plt.grid(axis='both')
        plt.legend(loc='best')
        plt.title(f"{seq} accuracy")
        plt.xlabel("After task")
        plt.ylabel("Accuracy [%]")
        if args.save_results:
            plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'acg_acc_test'))
        else:
            plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)
