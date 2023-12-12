#!/usr/bin python

import itertools
import subprocess

# /* TO MODIFY ---------------------------------------------

# SHIFT:
# shift_c_resnet50_size224.pth
# shift_resnet50_size224_seed1235.pth
# shift_resnet50_size224_seed1236.pth

# CLAD:
# clad_resnet50_size224.pth
# clad_resnet50_size224_seed1235.pth
# clad_resnet50_size224_seed1236.pth

dataset = 'clad' # clad, cifar10c, imagenetc, shift

usual_args = {}
if dataset == 'clad':
    usual_args['benchmark'] = ['clad']
    usual_args['pretrained_model_path'] = ['models_checkpoints/clad_resnet50_size224.pth']
    usual_args['model'] = ['resnet50']
elif dataset == 'cifar10c':
    usual_args['benchmark'] = ['cifar10c_standard']
    usual_args['model'] = ['wideresnet28']
elif dataset == 'imagenetc':
    usual_args['benchmark'] = ['imagenetc_standard']
    usual_args['model'] = ['resnet50']
elif dataset == 'shift':
    usual_args['benchmark'] = ['shift_mix_no_source']
    usual_args['pretrained_model_path'] = ['models_checkpoints/shift_c_resnet50_size224.pth']
    usual_args['model'] = ['resnet50']
    
# all params should be in a form of array, except EXP_NAME 
configs = {
    'ema_teacher': {
        'lr': [0.005, 0.00005],
        'run_name': 'atest'
    }
}

args_to_exp_name = ['lr']

common_args = '--data_root /datasets --save_results --cuda 0 --num_workers 5 --seeds 1234'

# TO MODIFY */ ---------------------------------------------

# add usual args to the arguments if they are not already in
for method in configs.keys():
    for arg in usual_args.keys():
        if arg not in configs[method].keys():
            configs[method][arg] = usual_args[arg]

for method, params in configs.items():

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
                    exp_name = param_name + str(val) + '_' + exp_name
                else:
                    exp_name = param_name + str(val)

        if len(exp_name) > 0:
            arguments += '--run_name' + ' ' + exp_name + ' '
            
        arguments += '--method' + ' ' + method + ' '
        arguments += '--dataset' + ' ' + dataset + ' '
            
        command = f"python test_adaptation.py {common_args}{arguments}"
        
        print(command)
        
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        for line in process.stdout:
            print(str(line, encoding='utf-8').strip())
        process.wait()
        if process.returncode == 1:
            print("--ERROR--" * 10)
