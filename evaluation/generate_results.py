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



LOGS_TO_USE = []
RESULTS_FOLDER = 'results'
LOGS_FOLDER = 'logs'
WINDOW_SIZE = 500 # for batch-wise train acc plot  

colors = {}

def get_plot_color(method):
    if method in colors.keys():
        return colors[method]

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
        if random_color not in colors.values() and 'white' not in random_color:
            break

    colors[method] = random_color
    return random_color

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_name', type=str, default=None, required=True,
                        help='')
    parser.add_argument('log_names', type=str, nargs='*',
                        help='name of the folder of log of the experiment')
    parser.add_argument('--save_results', action='store_true',
                        help='')
    parser.add_argument('--logs_regex', type=str,
                        help='')
    parser.add_argument('--per_class_acc', action='store_true',
                        help='')
    parser.add_argument('--pred_class_ratio', action='store_true',
                        help='')
    args = parser.parse_args()

    global LOGS_TO_USE
    LOGS_TO_USE = args.log_names

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
        
def plot_avgtime_avgacc(results, avg_accs_train, args):
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
        
def plot_acc_val(results, domains, args):
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
            plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'avg_acc_test'))
        else:
            plt.show()
        
def plot_acc_train(results, domains, args):
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
    plt.subplots_adjust(top=0.95)
    plt.title("Train sequences accuracy")
    plt.xlabel("Task")
    plt.ylabel("Accuracy [%]")
    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'train_acc_plot'))
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

        ax.set_title(method)
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

        ax.set_title(method)
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
    avg_accs = {'day': [], 'night': []}
    print("\nDay vs night results")
    for method in LOGS_TO_USE:
        sum_day, sum_night, num_day, num_night = 0, 0, 0, 0
        for domain, accs in zip(domains, results[method]["Top1_Acc_MB/train_phase/train_stream"]):
            # day for SHIFT, T1 and T3 are day sequences of CLAD
            if 'day' in domain or 'T1' in domain or 'T3' in domain:
                sum_day = np.sum(accs)
                num_day = len(accs)
                
            # night for SHIFT, T0, T2, T4 are day sequences of CLAD
            elif 'night' in domain or 'T0' in domain or 'T2' in domain or 'T4' in domain:
                sum_night = np.sum(accs)
                num_night = len(accs)

        avg_accs['day'].append((sum_day / num_day) * 100)
        avg_accs['night'].append((sum_night / num_night) * 100)
        
        print(f"- {method}: \tAcc day: {avg_accs['day'][-1]:.1f}, \tAcc night: {avg_accs['night'][-1]:.1f}")
        print(f"  {' ' * len(method)}  \tClass error day: {100 - avg_accs['day'][-1]:.1f}, \tClass error night: {100 - avg_accs['night'][-1]:.1f}")

    width = 0.3
    plt.xticks(range(len(LOGS_TO_USE)), LOGS_TO_USE, rotation=45)
    plt.title(f"Avg train accuracy day/night")
    ax.bar(np.array(range(len(LOGS_TO_USE))) - (width / 2), avg_accs['day'], color='gold', width=width, label='day')
    ax.bar(np.array(range(len(LOGS_TO_USE))) + (width / 2), avg_accs['night'], color='darkblue', width=width, label='night')
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy [%]")
    ax.grid(axis='both')
    plt.legend(loc='best')
    plt.tight_layout()

    if args.save_results:
        plt.savefig(os.path.join(RESULTS_FOLDER, args.results_name, 'avg_train_acc_and_time.png'))
    else:
        plt.show()

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
                   label=method,
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

    domains = get_and_check_domains()
    results = load_results()

    # =======================================================================================
    table = [["method", *domains, "Avg"], *[[] for i in range(len(LOGS_TO_USE))]]
    avg_accs_train = []
    per_domain_avg_accs_train = {method: [] for method in LOGS_TO_USE}
    for i, method in enumerate(LOGS_TO_USE):
        table[i + 1].append(method)
        flattened_results = []
        for j, single_domain in enumerate(results[method]["Top1_Acc_MB/train_phase/train_stream"]):

            # if domains[j] in domains[:j]:
            #     break
            
            single_domain_avg_acc = np.mean(single_domain) * 100.0
            per_domain_avg_accs_train[method].append(single_domain_avg_acc)
            table[i + 1].append(single_domain_avg_acc)
            flattened_results.extend(single_domain)

        avg_acc_train = np.mean(flattened_results) * 100.0

        print(f'- {method}: \tAcc: {avg_acc_train:.1f}, \tClassification error: {100 - avg_acc_train:.1f}')
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
    plot_acc_train(results, domains, args)
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
        plot_plot_per_timeofday_acc(results, domains, args)
    # # =======================================================================================
    # if 'frozen' in ''.join(LOGS_TO_USE):
    #     frozen_related_results_plot(per_domain_avg_accs_train, domains, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
