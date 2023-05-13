from .improved_wandb_logger import ImprovedWandBLogger
from .json_logger import JSONLogger
from utils.utils import get_experiment_name, get_experiment_folder

from avalanche.logging import TextLogger, InteractiveLogger, WandBLogger, CSVLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, amca_metrics, timing_metrics, EpochClassAccuracy
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import Module
import os


def get_eval_plugin(cfg, model: Module = None):
    loggers = [InteractiveLogger()]

    experiment_name = get_experiment_name(cfg)
    experiment_folder = get_experiment_folder(cfg)

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        
    path_to_log_file = os.path.join(experiment_folder, experiment_name + '_results')

    if cfg['save_results']:
        loggers.append(JSONLogger(cfg['num_classes'], open(path_to_log_file + '.json', 'w')))

    if cfg['wandb']:
        params = {'group': f"tta_{cfg['dataset']}", 'job_type': f"{cfg['method']}_{cfg['run_name']}"}
        if model is not None:
            wandb_logger = ImprovedWandBLogger(model=model, project_name=cfg['project_name'], run_name=cfg['run_name'], config=cfg, 
                                               params=params)
        else:
            wandb_logger = WandBLogger(project_name=cfg['project_name'], run_name=cfg['run_name'], config=cfg, params=params)
        loggers.append(wandb_logger)


    return EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch_running=True, stream=True),
        loss_metrics(epoch_running=True, stream=True),
        amca_metrics(streams=("test", "train", "val")),
        timing_metrics(epoch_running=True, epoch=True),
        EpochClassAccuracy(),
        loggers=loggers)
