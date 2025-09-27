#!/usr/bin python

import itertools
import subprocess
from copy import deepcopy
import os

# /* TO MODIFY ---------------------------------------------

CKPTS_DIR = 'models_checkpoints'

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
    'clad',
    # 'imagenetc',
    # 'shift',
    ] # clad, cifar10c, imagenetc, shift

run_name = 'TEST_CLEAR'


# all params should be in a form of array, except EXP_NAME 
configs = {
    'artta': {
        'run_name': run_name,
        # 'pretrained_model_path': ['models_checkpoints/clad_resnet50_size224_seed1236.pth']
    },
    # 'rmt': {
    #     'run_name': run_name,
    # },
    # 'source': {
    #     'run_name': run_name,
    #     'pretrained_model_path': ['models_checkpoints/clad_resnet50_size224_seed1236.pth']
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
    # 'bn_1': {
    #     'run_name': run_name,
    #     },
    # 'cotta': {
    #     'run_name': run_name,
    #     },
}

args_to_exp_name = [
    # 'batch_size',
    'lr',
    ]

common_args = '--data_root /datasets --save_results --cuda 0 --num_workers 5 \
    --seed 1234'
    # --seed 1235,1236'

# TO MODIFY */ ---------------------------------------------

def main():
    for dataset in datasets:
        perform_experiments(dataset)

def perform_experiments(dataset):
    usual_args = {}
    if dataset == 'clad':
        usual_args['pretrained_model_path'] = [os.path.join(CKPTS_DIR, 'clad_resnet50_size224.pth')]
        usual_args['model'] = ['resnet50']
    elif dataset == 'cifar10c':
        usual_args['model'] = ['wideresnet28']
    elif dataset == 'cifar10_1':
        usual_args['model'] = ['wideresnet28']
    elif dataset == 'imagenetc':
        usual_args['model'] = ['resnet50']
    elif dataset == 'shift':
        usual_args['pretrained_model_path'] = [os.path.join(CKPTS_DIR, 'shift_c_resnet50_size224.pth')]
        usual_args['model'] = ['resnet50']

    # add usual args to the arguments if they are not already in
    tmp_configs = deepcopy(configs)
    for method in configs.keys():
        for arg in usual_args.keys():
            if arg not in configs[method].keys():
                # TODO: set correct checkpoint depending on the seed
                
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
