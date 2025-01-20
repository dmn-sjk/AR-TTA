#!/usr/bin python

import itertools
import subprocess
from copy import deepcopy

# /* TO MODIFY ---------------------------------------------

# SHIFT:
# shift_c_resnet50_size224.pth
# shift_resnet50_size224_seed1235.pth
# shift_resnet50_size224_seed1236.pth

# CLAD:
# clad_resnet50_size224.pth``
# clad_resnet50_size224_seed1235.pth
# clad_resnet50_size224_seed1236.pth

datasets = [
    # 'cifar10c',
    # 'cifar10_1',
    # 'clad',
    'imagenetc',
    # 'shift',
    ] # clad, cifar10c, imagenetc, shift
# datasets = ['cifar10c']

# run_name = 'LoRA_ema'
# run_name = 'distancesMahaShrinkCovnorm_protoAllDomains'
# run_name = 'distancesEucl_protoAllDomains_normed'
# run_name = 'reset'
# run_name = 'test_maha'
# run_name = 'NoRep'
# run_name = 'plot'
run_name = 'TEST'


# all params should be in a form of array, except EXP_NAME 
configs = {
    # 'rmt': {
    #     'run_name': run_name,
    #     # 'batch_size': [10],
    #     # 'lr': [1e-3, 0.00025, 0.00003125],
    #     # 'bn_stats': [
    #     #     # 'dynamicbn'
    #     #     'source', 
    #     #     # 'test',
    #     #     ],
    # },
    # 'ema_teacher': {
    #     'run_name': run_name,
    #     'bn_stats': ['dynamicbn'],
    #     # 'bn_dist_scale': [100],
    #     'batch_size': [10],
    #     # 'lora': [''],
    #     # 'lora_rank': [4],
    #     # 'rank_mode': [
    #     #     # 'percentile', 
    #     #     # 'ratio',
    #     #     # 'threshold',
    #     #     'fixed'
    #     #     ]
    # },
    # 'nonparametric': {
    #     'run_name': run_name,
    #     'bn_stats': [
    #         # 'dynamicbn'
    #         # 'source', 
    #         'test',
    #         ],
    #     # 'bn_dist_scale': [100],
    #     'batch_size': [10],
    # },
    # 'custom': {
    #     'run_name': run_name,
    #     # 'bn_stats': [
    #     #     # 'dynamicbn',
    #     #     'source',
    #     #     # 'test',
    #     #     ],
    #     'bn_dist_scale': [
    #         0.1,
    #         # 1.0, 
    #         # 10.0
    #         ],
    #     'batch_size': [10],
    #     'memory_size': [2000],
    #     # 'replay_augs': [
    #     #     # 'cotta',
    #     #     # 'mixup_from_memory',
    #     #     'null',
    #     #     ],
    #     'lr': [1e-3, 0.00025, 0.00003125],
    #     # 'update_method': ['source_pseudolabels'],
    # },
    # 'frozen': {
    #     'run_name': run_name,
    #     },
    # 'eata': {
    #     'run_name': run_name,
    #     },
    # 'tent': {
    #     'run_name': run_name,
    #     },
    # 'sar': {
    #     'run_name': run_name,
    #     },
    # # 'bn_stats_adapt': {'run_name': run_name,},
    
    'custom': {
        'run_name': run_name,
        'memory_size': [0],
        'replay_augs': ['null']
    },
    # 'cotta': {
    #     'run_name': run_name,
    #     },
}

args_to_exp_name = [
    'bn_stats',
    # 'replay_augs',
    # 'lora_rank',
    # 'rank_mode'
    # 'batch_size',
    # 'update_method',
    'replay_augs',
    'lr',
    'bn_dist_scale',
    ]

common_args = '--data_root /datasets --save_results --cuda 0 --num_workers 5 --seeds 1234'

# TO MODIFY */ ---------------------------------------------

def main():
    for dataset in datasets:
        perform_experiments(dataset)

def perform_experiments(dataset):
    usual_args = {}
    if dataset == 'clad':
        usual_args['benchmark'] = ['clad']
        usual_args['pretrained_model_path'] = ['models_checkpoints/clad_resnet50_size224.pth']
        usual_args['model'] = ['resnet50']
    elif dataset == 'cifar10c':
        usual_args['benchmark'] = ['cifar10c_standard']
        usual_args['model'] = ['wideresnet28']
    elif dataset == 'cifar10_1':
        usual_args['benchmark'] = ['cifar10_1_standard']
        usual_args['model'] = ['wideresnet28']
    elif dataset == 'imagenetc':
        usual_args['benchmark'] = ['imagenetc_standard_subset']
        usual_args['model'] = ['resnet50']
    elif dataset == 'shift':
        usual_args['benchmark'] = ['shift_mix_no_source']
        usual_args['pretrained_model_path'] = ['models_checkpoints/shift_c_resnet50_size224.pth']
        usual_args['model'] = ['resnet50']

    # add usual args to the arguments if they are not already in
    tmp_configs = deepcopy(configs)
    for method in configs.keys():
        for arg in usual_args.keys():
            if arg not in configs[method].keys():
                tmp_configs[method][arg] = usual_args[arg]

    for method, params in tmp_configs.items():

        base_exp_name = ""
        if "run_name" in params.keys():
            base_exp_name = params["run_name"]
            del params["run_name"]
        
        for params_vals in itertools.product(*params.values()):
            exp_name = base_exp_name
            arguments = ' '
            for param_name, val in zip(params.keys(), params_vals):
                arguments += '--' + param_name + ' ' + str(val) + ' '

                if param_name in args_to_exp_name:
                    if len(exp_name) != 0:
                        exp_name = param_name.upper() + str(val) + '_' + exp_name
                    else:
                        exp_name = param_name.upper() + str(val)

            if len(exp_name) > 0:
                arguments += '--run_name' + ' ' + exp_name + ' '
                
            arguments += '--method' + ' ' + method + ' '
            arguments += '--dataset' + ' ' + dataset + ' '
                
            command = f"python test_adaptation.py {common_args}{arguments}"
            
            print(command)
            
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in process.stdout:
                print(str(line, encoding='utf-8').strip())
            for line in process.stderr:
                print(str(line, encoding='utf-8').strip())
            process.wait()
            if process.returncode == 1:
                print("--ERROR--" * 10)
                
if __name__ == "__main__":
    main()
