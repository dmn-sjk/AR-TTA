#!/usr/bin python

import select
import itertools
import subprocess
from copy import deepcopy
import os
import time

# /* TO MODIFY ---------------------------------------------

CKPTS_DIR = 'models_checkpoints'
DATA_ROOT = '/datasets'

# SLURM
SLURM = False
RUN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = 'slurm_jobs'
CPUS_PER_TASK = 10
MEM_PER_CPU = 3
GPUS = 1
ACCOUNT = ""
PARTITION = ""
GRES = ""


datasets = [
    'cifar10c',
    # 'cifar10_1',
    # 'clad',
    # 'imagenetc',
    # 'shift',
    ]

seeds = [
    1234,
    # 1235,
    # 1236
]

RUN_NAME = ''


# all params should be in a form of array, except EXP_NAME 
configs = {
    # 'source': {
    #     'run_name': RUN_NAME,
    #     },
    'artta': {
        'run_name': RUN_NAME,
        'memory_size': [0, 2000],
    },
    # 'rmt': {
    #     'run_name': RUN_NAME,
    # },
    # 'eata': {
    #     'run_name': RUN_NAME,
    #     },
    # 'tent': {
    #     'run_name': RUN_NAME,
    #     },
    # 'sar': {
    #     'run_name': RUN_NAME,
    #     },
    # 'bn_1': {
    #     'run_name': RUN_NAME,
    #     },
    # 'cotta': {
    #     'run_name': RUN_NAME,
    #     },
}

# args to include in experiment name
args_to_exp_name = [
    'memory_size'
]

common_args = f'--data_root {DATA_ROOT} --ckpts_dir {CKPTS_DIR}'

# TO MODIFY */ ---------------------------------------------

def main():
    if SLURM:
        if not os.path.exists(JOBS_DIR):
            os.mkdir(JOBS_DIR)

    for dataset in datasets:
        for seed in seeds:
            perform_experiments(dataset, seed)

def add_typical_args(configs, dataset, seed):
    _configs = deepcopy(configs)
    # add typical values of arguments, overwrite it if the configs changes the value
    for method, config in configs.items():
        typical_config = {
            'model': [get_model(dataset)],
            'src_model_ckpt_file': [get_ckpt_file(dataset, seed)]
        }
        typical_config.update(config)
        _configs[method] = typical_config
    return _configs

def get_ckpt_file(dataset: str, seed: int):
    if dataset in ['cifar10c', 'cifar10_1', 'imagenetc']:
        # ckpts for those datasets are handled by robustbench library
        return None
    elif dataset == 'shift':
        if seed == 1235:
            return 'shift_resnet50_size224_seed1235.pth'
        elif seed == 1236:
            return 'shift_resnet50_size224_seed1236.pth'
        else:
            return 'shift_c_resnet50_size224.pth'
    elif dataset == 'clad':
        if seed == 1235:
            return 'clad_resnet50_size224_seed1235.pth'
        elif seed == 1236:
            return 'clad_resnet50_size224_seed1236.pth'
        else:
            return 'clad_resnet50_size224.pth'
    else:
        raise NotImplementedError(dataset)
    
def get_model(dataset):
    if dataset in ['clad', 'imagenetc', 'shift']:
        return 'resnet50'
    elif dataset in ['cifar10c', 'cifar10_1']:
        return 'wideresnet28'
    else:
        raise NotImplementedError(dataset)
    
def run_slurm(command, run_name):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    job_folder = os.path.join(JOBS_DIR, run_name + '_' +  current_time)
    if not os.path.exists(job_folder):
        os.mkdir(job_folder)

    job_file = os.path.join(job_folder, f"{run_name}.job")
    output_file = os.path.join(job_folder, f"{run_name}.out")
    error_file = os.path.join(job_folder, f"{run_name}.err")
    
    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={run_name}.job\n")
        fh.writelines(f"#SBATCH --output={output_file}\n")
        fh.writelines(f"#SBATCH --error={error_file}\n")
        fh.writelines(f"#SBATCH --gpus={GPUS}\n")
        fh.writelines(f"#SBATCH --account={ACCOUNT}\n")
        fh.writelines(f"#SBATCH --partition={PARTITION}\n")
        fh.writelines(f"#SBATCH --gres={GRES}\n")
        fh.writelines(f"#SBATCH --cpus-per-task={CPUS_PER_TASK}\n")
        fh.writelines(f"#SBATCH --mem-per-cpu={MEM_PER_CPU}G\n")


        fh.writelines(f"cd {RUN_SCRIPT_DIR}\n")
        fh.writelines("module load Miniconda3/4.9.2\n")
        fh.writelines("eval \"$(conda shell.bash hook)\"\n")
        fh.writelines("conda activate artta\n")

        fh.writelines(command)
        fh.writelines("\n")

    os.system(f"sbatch {job_file}")
    
def run_normal(command):
    # Force unbuffered output by prepending "python -u"
    if command.startswith("python "):
        command = "python -u" + command[6:]
        
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        bufsize=0,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
    )
 
    # Get file descriptors for stdout and stderr
    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    
    while True:
        reads = [stdout_fd, stderr_fd]
        ret = select.select(reads, [], [])
        
        for fd in ret[0]:
            if fd == stdout_fd:
                line = process.stdout.readline()
                if line:
                    print(line.decode().rstrip(), flush=True)
            if fd == stderr_fd:
                line = process.stderr.readline()
                if line:
                    print(line.decode().rstrip(), flush=True)
        
        if process.poll() is not None:
            # Read any remaining output
            for line in process.stdout:
                print(line.decode().rstrip(), flush=True)
            for line in process.stderr:
                print(line.decode().rstrip(), flush=True)
            break
        
    if process.returncode != 0:
        print("--ERROR--" * 20)
    
def run_command(command, run_name):
    if SLURM:
        run_slurm(command, run_name)
    else:
        run_normal(command)
            
def perform_experiments(dataset, seed):
    tmp_configs = add_typical_args(configs, dataset, seed)

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
            arguments += '--seed' + ' ' + str(seed) + ' '
                
            command = f"python test_adaptation.py {common_args}{arguments}"
            
            print(command)

            run_command(command, exp_name)
            
if __name__ == "__main__":
    main()
