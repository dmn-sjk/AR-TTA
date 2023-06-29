import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import deque
import argparse
import csv
import shutil
import yaml
import matplotlib.colors as mcolors
import glob
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager

sns.set_theme()
sns.set_context("paper")

font_legend = font_manager.FontProperties(
                                   style='normal', size=12)

font = {'weight': 'normal',
        'size': 12,
        }


LOGS_TO_USE = []
RESULTS_FOLDER = 'results'
LOGS_FOLDER = 'logs'
WINDOW_SIZE = 100 # for batch-wise train acc plot
DISCARD_REPEATED_DOMAINS = False
NOT_INCLUDE_LAST_DOMAIN_IN_AVERAGE = True
OLD_CLAD_DOMAIN_NAMES = False
POSSIBLE_METHODS = ['cotta', 'eata', 'tent', 'frozen', 'finetune', 'sar', 'custom', 'bn_stats_adapt']
LABELS = ['CoTTA', 'EATA', 'TENT', 'Source', 'finetune', 'SAR', 'AR-TTA (ours)', 'BN stats adapt']
USELESS_COLORS = ['gainsboro', 'snow', 'mistyrose', 'seashell', 'linen', 'oldlace',
                  'comsilk', 'ivory', 'lightyellow', 'honeydew', 'azure', 'aliceblue',
                  'lavender', 'lavenderblush', 'mintcream']

colors = {}


def get_label(log_name):
    for i, method in enumerate(POSSIBLE_METHODS):
        start_idx = log_name.find(method)
        if start_idx != -1:
            # return LABELS[i]
            return log_name[start_idx:]
    raise ValueError(f"No method name found in log name: {log_name}")

def get_plot_color(method):
    if method in colors.keys():
        return colors[method]
    
    if 'custom' in method:
        return 'tab:green'

    if 'frozen' in method and 'tab:red' not in colors.values():
        colors[method] = 'tab:red'
        return 'tab:red'
    elif 'finetune' in method and 'tab:green' not in colors.values():
        colors[method] = 'tab:green'
        return 'tab:green'
    elif 'eata' in method and 'tab:orange' not in colors.values():
        colors[method] = 'tab:orange'
        return 'tab:orange' 
    
    # COTTA mods
    # elif 'no_stochastic_res' in method:
    #     colors[method] = 'greenyellow'
    #     return 'greenyellow'
    # elif 'distilltemp4_out' in method:
    #     colors[method] = 'blueviolet'
    #     return 'blueviolet'
    # elif 'distilltemp2_out' in method:
    #     colors[method] = 'deepskyblue'
    #     return 'deepskyblue'
    # elif 'distill_feat_with_loss' in method:
    #     colors[method] = 'deeppink'
    #     return 'deeppink'
    # elif 'distill_feat_without_loss' in method:
    #     colors[method] = 'teal'
    #     return 'teal'
    # elif 'no_teacher' in method:
    #     colors[method] = 'peru'
    #     return 'peru'
    
    elif 'cotta' in method and 'tab:blue' not in colors.values():
        colors[method] = 'tab:blue'
        return 'tab:blue' 
    elif 'tent' in method and 'tab:purple' not in colors.values():
        colors[method] = 'tab:purple'
        return 'tab:purple'
    
    # SAR mods
    # elif 'unifoptim_nores' in method:
    #     colors[method] = 'greenyellow'
    #     return 'greenyellow'  
    # elif 'sar_setup_nores' in method:
    #     colors[method] = 'blueviolet'
    #     return 'blueviolet'
    # elif 'sar_setup' in method:
    #     colors[method] = 'deepskyblue'
    #     return 'deepskyblue'
    # elif 'frozen_resnet50gn_gn' in method:
    #     colors[method] = 'deeppink'
    #     return 'deeppink' 
    
    
    elif 'sar' in method and 'tab:brown' not in colors.values():
        colors[method] = 'tab:brown'
        return 'tab:brown' 


    while True:
        random_color = np.random.choice(list(mcolors.CSS4_COLORS))
        if random_color not in colors.values() \
            and random_color not in USELESS_COLORS \
            and 'grey' not in USELESS_COLORS \
            and 'white' not in random_color:
            break

    colors[method] = random_color
    return random_color

def parse_args():
    parser = argparse.ArgumentParser(description=
                                     'Code for generating results of TTA based on logs saved in logs folder during TTA via --save_results command line flag. Genrated results are saved in results folder.')
    parser.add_argument('--results_name', type=str, default=None, required=True,
                        help='Name of the folder in which results will be saved. ')
    parser.add_argument('log_names', type=str, nargs='*',
                        help='Names of the folders of logs of the chosen experiments.')
    parser.add_argument('--save_results', action='store_true',
                        help='Save plots and avg accuracies, instead of displaying them.')
    parser.add_argument('--logs_regex', type=str,
                        help='Regex expression for matching folders of experiments logs to generate results from.')
    parser.add_argument('--per_class_acc', action='store_true',
                        help='Plot per class accuracy.')
    parser.add_argument('--pred_class_ratio', action='store_true',
                        help='Plot ratio of predicitons vs occurances of each class.')
    args = parser.parse_args()

    global LOGS_TO_USE
    LOGS_TO_USE = args.log_names

    if args.logs_regex is not None:
        regex_matches = [log.split('/')[-1] for log in glob.glob(os.path.join('logs', args.logs_regex))]
        print(f"Num of regex matches: {len(regex_matches)}")
        LOGS_TO_USE.extend(regex_matches)
        
    if len(LOGS_TO_USE) == 0:
        raise ValueError("Provide log folders' names to generate results from") 
        
    return args

def get_and_check_domains():
    domains = []
    for log in LOGS_TO_USE:
        with open(os.path.join(LOGS_FOLDER, log, log + '_config.yaml'), "r") as f:
            curr_domains = yaml.load(f, Loader=yaml.Loader)['domains']
            
            if len(curr_domains) == 0:
                raise ValueError("Empty domains list!")
            
            if len(domains) == 0:
                domains = curr_domains
                
            if domains != curr_domains:
                raise ValueError(f"Every used exeperiment should have the same domains in the same order:\n \
                                 - domains from previous logs: {domains}\n \
                                 - domains in {log} logs folder: {curr_domains}")
    
    if DISCARD_REPEATED_DOMAINS:
        modified_domains = []
        accepted_domains_idxs = []
        for i, domain in enumerate(domains):
            if domain in domains[:i]:
                continue
            accepted_domains_idxs.append(i)
            modified_domains.append(domain)
            
        return modified_domains, accepted_domains_idxs
    else:
        return domains, list(range(len(domains)))

def load_results(accepted_domains_idxs):
    results = {}

    for log in LOGS_TO_USE:
        with open(os.path.join(LOGS_FOLDER, log, log + '_results.json'), 'r') as f:
            results[log] = json.load(f)
            
            if DISCARD_REPEATED_DOMAINS:
                for key in results[log].keys():

                    modified_results = []
                    
                    # test_stream have one more value because of initial eval before TTA
                    if 'test_stream' in key:
                        modified_results.append(results[log][key][0])
                        
                    for idx in accepted_domains_idxs:
                        if 'test_stream' in key:
                            modified_results.append(results[log][key][idx + 1])
                        else:
                            modified_results.append(results[log][key][idx])

                    results[log][key] = modified_results

    return results

def copy_config_files():
    for log in LOGS_TO_USE:
        shutil.copy(os.path.join(LOGS_FOLDER, log, log + '_config.yaml'),
                    os.path.join(RESULTS_FOLDER, args.results_name))
        
def plot_avgtime_avgacc(results, avg_accs_train, args):
    fig, ax = plt.subplots()
    # avg_accs = []
    avg_times = []
    for i, method in enumerate(LOGS_TO_USE):
        # avg_accs.append(avg_accs_train[i])
        avg_times.append(np.mean(results[method]["AvgTimeIter/train_phase/train_stream"]))

    width = 0.3
    plt.xticks(range(len(avg_accs_train)), [get_label(log) for log in LOGS_TO_USE], rotation=45)
    plt.title(f"Avg train accuracy and run time")
    ax.bar(np.array(range(len(avg_accs_train))) - (width / 2), avg_accs_train, color='blue', width=width)
    ax2 = ax.twinx()
    ax2.bar(np.array(range(len(avg_accs_train))) + (width / 2), avg_times, color='red', width=width)
    ax2.set_ylabel('Avg run time', color='red')
    plt.xlabel("After task")
    ax.set_ylabel("Accuracy [%]", color='blue')
    plt.tight_layout()

    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'avg_train_acc_and_time.png'))
    else:
        plt.show()
        
def plot_acc_val(results, domains, args):
    # checking only test stream
    seqs = ["Top1_Acc_Stream/eval_phase/test_stream"]
    for seq in seqs:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for method in LOGS_TO_USE:
            accs = np.array(results[method][seq]) * 100.0
            ax.plot(range(len(accs)), accs, marker='.', label=get_label(method), color=get_plot_color(method))
            print(f"TEST {method}: {accs[0]}")

        plt.xticks(range(len(domains) + 1), ['Init', *range(len(domains))])
        plt.grid(axis='both')
        plt.legend(loc='best')
        plt.title(f"{seq} accuracy")
        plt.xlabel("After task")
        plt.ylabel("Accuracy [%]")
        plt.tight_layout()
        if args.save_results:
            plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'avg_acc_test'))
        else:
            plt.show()
            
def plot_domainwise_acc_train(results, domains, args):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    for method in LOGS_TO_USE:
        accs = []
        for domain_res in results[method]["Top1_Acc_MB/train_phase/train_stream"]:
            accs.append(np.mean(domain_res) * 100.0)
        ax.plot(range(len(accs)), accs, marker='.', label=get_label(method), color=get_plot_color(method))

    plt.xticks(range(len(domains)), domains, rotation=45)
    plt.grid(axis='both')
    plt.legend(loc='best')
    plt.title(f"TTA accuracy")
    plt.xlabel("Domain")
    plt.ylabel("Accuracy [%]")
    plt.tight_layout()
    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'train_domainwise_acc_plot'))
    else:
        plt.show()
    
def plot_batchwise_acc_train(results, domains, args):
    window = deque(maxlen=WINDOW_SIZE)

    # TRAINING SEQUENCES
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in LOGS_TO_USE:
        whole_results = []
        window.clear()
        for i, task_results in enumerate(results[method]["Top1_Acc_MB/train_phase/train_stream"]):
            # if i > 4:
            #     break
                
            window_accs = []
            window.clear()
            for batch_acc in task_results:
                window.append(batch_acc)
                window_accs.append(np.mean(window))
            whole_results.extend(window_accs)
        accs = np.array(whole_results) * 100.0
        if 'frozen'in method:
            ax.plot(range(len(whole_results)), accs, '--', label=get_label(method), color=get_plot_color(method))
        else:
            ax.plot(range(len(whole_results)), accs, '-', label=get_label(method), color=get_plot_color(method))

    end_of_x_axis = 0
    xticks = [end_of_x_axis]
    for i, task_results in enumerate(results[LOGS_TO_USE[0]]["Top1_Acc_MB/train_phase/train_stream"]):
        # if i > 4:
        #     break
        task_samples = len(task_results)
        end_of_x_axis += task_samples
        xticks.append(end_of_x_axis)

    plt.xticks(xticks, [*domains, ''], rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(axis='both')
    plt.legend(loc='best', prop=font_legend)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.95)
    # plt.title("Train sequences accuracy")
    plt.xlabel("Task", fontdict=font)
    plt.ylabel("Accuracy [%]", fontdict=font)
    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'train_batchwise_acc_plot'))
    else:
        plt.show()
        
def plot_per_class_acc(results, domains, args):
    result_key = "Top1_ClassAcc_Epoch/train_phase/train_stream/"
    
    logs_with_per_class_acc_data = []
    for method in LOGS_TO_USE:
        check_class_id = 0
        if result_key + str(check_class_id) in results[method].keys():
            logs_with_per_class_acc_data.append(method)
        else:
            print(f"No per class accuracy data in for {method}!")
    
    max_cols = 3
    nrows = int(np.ceil(len(logs_with_per_class_acc_data) / max_cols))
    ncols = max_cols if len(logs_with_per_class_acc_data) >= max_cols else len(logs_with_per_class_acc_data)
    fig, axs = plt.subplots(nrows=nrows,
                            ncols=ncols, figsize=(15, 10))

    for i, method in enumerate(logs_with_per_class_acc_data):
        if nrows > 1 and ncols > 1:  
            row = i // max_cols
            col = i % max_cols
            ax = axs[row, col]
        elif ncols == 1:
            ax = axs
        else:
            col = i % max_cols
            ax = axs[col]
        
        class_id = 0
        # while True, because of lack of information about class num
        while True:
            if result_key + str(class_id) not in results[method].keys():
                break

                
            accs_array = np.array(results[method][result_key + str(class_id)])

            # multiply only elements which are not None 
            # (in CLAD not all classes are in every sequence so the acc is none)
            accs = np.multiply(accs_array, np.full_like(accs_array, 100), where=accs_array!=None)
            ax.plot(range(len(accs)), accs, marker='.', label=class_id)

            class_id += 1

        ax.set_title(get_label(method))
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains)
        ax.grid(axis='both')
        ax.set_xlabel("Domain")
        ax.set_ylabel("Accuracy [%]")
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')
        
        fig.suptitle("Per class accuracy", fontsize="x-large")

    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'per_class_acc'))
    else:
        plt.show()
        
def plot_pred_class_ratio(results, domains, args):
    per_class_samples_key = "per_class_samples"
    per_class_predictions_key = "per_class_predictions"
    
    logs_with_per_class_data = []
    for method in LOGS_TO_USE:
        if per_class_samples_key in results[method].keys():
            logs_with_per_class_data.append(method)
        else:
            print(f"No per_class_samples and per_class_predictions data for {method}!")
    
    max_cols = 3
    nrows = int(np.ceil(len(logs_with_per_class_data) / max_cols))
    ncols = max_cols if len(logs_with_per_class_data) >= max_cols else len(logs_with_per_class_data)
    fig, axs = plt.subplots(nrows=nrows,
                            ncols=ncols, figsize=(15, 10))

    max_y_val = -1e6
    min_y_val = 1e6
    for i, method in enumerate(logs_with_per_class_data):
        if nrows > 1 and ncols > 1:  
            row = i // max_cols
            col = i % max_cols
            ax = axs[row, col]
        elif ncols == 1:
            ax = axs
        else:
            col = i % max_cols
            ax = axs[col]
        
        per_class_samples = np.array(results[method][per_class_samples_key])
        per_class_predictions = np.array(results[method][per_class_predictions_key])
        preds_samples_ratio = np.divide(per_class_predictions, per_class_samples, 
                                        out=np.full_like(per_class_predictions, fill_value=None,),
                                        where=per_class_samples != 0)
        curr_max_val = np.nanmax(preds_samples_ratio)
        curr_min_val = np.nanmin(preds_samples_ratio)
        if curr_max_val > max_y_val: max_y_val = curr_max_val
        if curr_min_val < min_y_val: min_y_val = curr_min_val
        
        num_classes = len(per_class_samples[0])

        for class_id in range(num_classes):
            ax.plot(range(len(domains)), preds_samples_ratio[:, class_id],
                               marker='.', label=class_id)

            class_id += 1

        ax.set_title(get_label(method))
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains)
        ax.grid(axis='both')
        ax.set_xlabel("Domain")
        ax.set_ylabel("Num of predictions to num of samples ratio")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')
        
        fig.suptitle("Per class preds vs samples ratio", fontsize="x-large")
    
    for ax in axs.reshape(-1):
        ax.set_ylim(min_y_val, max_y_val)

    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'per_class_preds_samples_ratio'))
    else:
        plt.show()
        
def plot_plot_per_timeofday_acc(results, domains, args):
    fig, ax = plt.subplots(figsize=(10, 7))
    avg_accs = {'day': [], 'dawn/dusk': [], 'night': []}
    min_acc_val = 100

    print("\nTime of day results")
    for method in LOGS_TO_USE:
        sum_day, sum_night, sum_dawndusk, num_day, num_night, num_dawndusk = 0, 0, 0, 0, 0, 0
        for i, (domain, accs) in enumerate(zip(domains, results[method]["Top1_Acc_MB/train_phase/train_stream"])):
            
            if NOT_INCLUDE_LAST_DOMAIN_IN_AVERAGE and i == len(domains) - 1:
                break

            if OLD_CLAD_DOMAIN_NAMES:
                clad_day_domains = ['T1', 'T3']
                clad_night_domains = ['T0', 'T2', 'T4']
            else:
                clad_day_domains = ['T2', 'T4']
                clad_night_domains = ['T1', 'T3', 'T5']
            
            # day for SHIFT, T1 and T3 are day sequences of CLAD (old numbering)
            if 'day' in domain or any([day_domain in domain for day_domain in clad_day_domains]):
                sum_day += np.sum(accs)
                num_day += len(accs)

            # night for SHIFT, T0, T2, T4 are day sequences of CLAD (old numbering)
            elif 'night' in domain or any([night_domain in domain for night_domain in clad_night_domains]):
                sum_night += np.sum(accs)
                num_night += len(accs)
            
            elif 'dawn/dusk' in domain:
                sum_dawndusk += np.sum(accs)
                num_dawndusk += len(accs)

        acc_day, acc_night, acc_dawndusk = 100, 100, 100
        if num_day != 0:
            acc_day = (sum_day / num_day) * 100
            avg_accs['day'].append(acc_day)
        if num_night != 0:
            acc_night = (sum_night / num_night) * 100
            avg_accs['night'].append(acc_night)
        if num_dawndusk != 0:
            acc_dawndusk = (sum_dawndusk / num_dawndusk) * 100
            avg_accs['dawn/dusk'].append(acc_dawndusk)

        min_acc_val = np.min([acc_day, acc_night, acc_dawndusk, min_acc_val])
        
        to_print_acc = f"- {method}: "
        to_print_err = f"  {' ' * len(method)} "
        for timeofday in avg_accs.keys():
            if len(avg_accs[timeofday]) > 0:
                to_print_acc += f"\tAcc {timeofday}: {avg_accs[timeofday][-1]:.1f} "
                to_print_err += f"\tClass error {timeofday}: {100 - avg_accs[timeofday][-1]:.1f} "
                
        print(to_print_acc)
        print(to_print_err)
        
    width = 0.25
    plt.xticks(range(len(LOGS_TO_USE)), [get_label(log) for log in LOGS_TO_USE], rotation=45)
    plt.title(f"Avg train accuracy time of day")
    if len(avg_accs['dawn/dusk']) > 0:
        ax.bar(np.array(range(len(LOGS_TO_USE))) - width, avg_accs['day'], color='gold', width=width, label='day')
        ax.bar(np.array(range(len(LOGS_TO_USE))), avg_accs['dawn/dusk'], color='orange', width=width, label='dawn/dusk')
        ax.bar(np.array(range(len(LOGS_TO_USE))) + width, avg_accs['night'], color='darkblue', width=width, label='night')
    else:
        ax.bar(np.array(range(len(LOGS_TO_USE))) - (width / 2), avg_accs['day'], color='gold', width=width, label='day')
        ax.bar(np.array(range(len(LOGS_TO_USE))) + (width / 2), avg_accs['night'], color='darkblue', width=width, label='night')
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(min_acc_val - 5, 100)
    ax.grid(axis='both')
    plt.legend(loc='best')
    plt.tight_layout()

    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'timeofday_acc.png'))
    else:
        plt.show()
        
    return avg_accs

def frozen_related_results_plot(per_domain_avg_accs_train, domains, args):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    frozen_method_log = ''
    for method in LOGS_TO_USE:
        if 'frozen' in method:
            frozen_method_log = method
    
    ys = []
    for method in LOGS_TO_USE:
        if method == frozen_method_log:
            continue
        
        subtraction = np.subtract(per_domain_avg_accs_train[method],
                                  per_domain_avg_accs_train[frozen_method_log])
        ax.scatter(per_domain_avg_accs_train[frozen_method_log],
                   subtraction,
                   label=get_label(method),
                   color=get_plot_color(method))
        ys.extend(subtraction)
        
    xs = per_domain_avg_accs_train[frozen_method_log] * (len(LOGS_TO_USE) - 1)
    # calculate equation for trendline
    z = np.polyfit(xs, ys, 1)
    p = np.poly1d(z)

    # add trendline to plot
    ax.plot(xs, p(xs), 'y--', label='Trend')

    ax.set_title("Frozen-related accuracy")
    ax.set_xlabel("Frozen avg acc [%]")
    ax.set_ylabel("method avg acc - frozen avg acc [%]")
    ax.grid(axis='both')
    plt.legend(loc='best')
    plt.tight_layout()
    
    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'frozen_related_results.png'))
    else:
        plt.show()
    
        
def main(args):
    if args.save_results:
        os.makedirs(os.path.join(RESULTS_FOLDER, args.results_name), exist_ok=True)
    
    if args.save_results:
        copy_config_files()

    domains, accepted_domains_idxs = get_and_check_domains()
    results = load_results(accepted_domains_idxs)
    
    # =======================================================================================
    table = [["method", *domains, "Avg"], *[[] for i in range(len(LOGS_TO_USE))]]
    avg_accs_train = []
    per_domain_avg_accs_train = {method: [] for method in LOGS_TO_USE}
    for i, method in enumerate(LOGS_TO_USE):
        table[i + 1].append(method)
        flattened_results = []
        for j, single_domain in enumerate(results[method]["Top1_Acc_MB/train_phase/train_stream"]):
            single_domain_avg_acc = np.mean(single_domain) * 100.0
            per_domain_avg_accs_train[method].append(single_domain_avg_acc)
            table[i + 1].append(single_domain_avg_acc)

            if not (NOT_INCLUDE_LAST_DOMAIN_IN_AVERAGE and j == len(results[method]["Top1_Acc_MB/train_phase/train_stream"]) - 1):
                flattened_results.extend(single_domain)

        avg_acc_train = np.mean(flattened_results) * 100.0

        # whether to print source domain result as the last domain
        if domains[-1] in ['clear', 'daytime_clear', 'T0']:
            print(f'- {method}: \tAcc: {avg_acc_train:.2f}, \tClassification error: {100 - avg_acc_train:.2f}, \tSource acc: {single_domain_avg_acc:.2f}')
        else:
            print(f'- {method}: \tAcc: {avg_acc_train:.2f}, \tClassification error: {100 - avg_acc_train:.2f}')

        table[i + 1].append(avg_acc_train)
        
        avg_accs_train.append(avg_acc_train)

    if args.save_results:
        with open(os.path.join(RESULTS_FOLDER, args.results_name, args.results_name + '.csv'), 'w') as f:
            writer = csv.writer(f)
            for row in table:
                writer.writerow(row)
    # =======================================================================================
    plot_avgtime_avgacc(results, avg_accs_train, args)
    # =======================================================================================
    plot_batchwise_acc_train(results, domains, args)
    # =======================================================================================    
    plot_domainwise_acc_train(results, domains, args)
    # =======================================================================================
    plot_acc_val(results, domains, args)
    # =======================================================================================
    if args.per_class_acc:
        plot_per_class_acc(results, domains, args)
    # =======================================================================================
    if args.pred_class_ratio:
        plot_pred_class_ratio(results, domains, args)
    # =======================================================================================
    if 'shift' in LOGS_TO_USE[0] or 'clad' in LOGS_TO_USE[0]:
        per_time_of_day_accs = plot_plot_per_timeofday_acc(results, domains, args)
    # =======================================================================================
    # if 'frozen' in ''.join(LOGS_TO_USE):
    #     frozen_related_results_plot(per_domain_avg_accs_train, domains, args)
    
    # accs = np.array(list(per_time_of_day_accs.values())).T
    
    # fig, ax = plt.subplots(figsize=(10, 7))
    
    # for i, method in enumerate(LOGS_TO_USE):
    #     if 'frozen' in method:
    #         frozen_method_log_idx = i
    
    # ys = []
    # for i, method in enumerate(LOGS_TO_USE):
    #     if i == frozen_method_log_idx or 'finetune' in method:
    #         continue
        
    #     subtraction = np.subtract(accs[i],
    #                               accs[frozen_method_log_idx])
    #     ax.scatter(accs[frozen_method_log_idx],
    #                subtraction,
    #                label=get_label(method),
    #                color=get_plot_color(method))
    #     ys.extend(subtraction)

    # xs = list(accs[frozen_method_log_idx]) * (len(LOGS_TO_USE) - 2)
    # # calculate equation for trendline
    # z = np.polyfit(xs, ys, 1)
    # p = np.poly1d(z)

    # # add trendline to plot
    # ax.plot(xs, p(xs), 'y--', label='Trend')

    # ax.set_title("Frozen-related accuracy")
    # ax.set_xlabel("Frozen avg acc [%]")
    # ax.set_ylabel("method avg acc - frozen avg acc [%]")
    # ax.grid(axis='both')
    # plt.legend(loc='best')
    # plt.tight_layout()
    
    # if args.save_results:
    #     plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'frozen_related_results.png'))
    # else:
    #     plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
