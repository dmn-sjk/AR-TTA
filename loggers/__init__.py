from .improved_wandb_logger import ImprovedWandBLogger

from avalanche.logging import TextLogger, InteractiveLogger, WandBLogger, CSVLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, amca_metrics, timing_metrics
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import Module
import os


def get_eval_plugin(cfg, model: Module = None):
    loggers = [InteractiveLogger()]

    experiment_name = f"{cfg['dataset']}_{cfg['benchmark']}_{cfg['method']}_{cfg['model']}_{cfg['run_name']}"
    experiment_folder = os.path.join(cfg['log_dir'], experiment_name)

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    if cfg['text_logger']:
        loggers.append(TextLogger(open(f"{experiment_folder}/{experiment_name}.txt", 'w')))

    if cfg['save_results']:
        loggers.append(CSVLogger(experiment_folder))

    if cfg['wandb']:
        if model is not None:
            wandb_logger = ImprovedWandBLogger(model=model, project_name=cfg['project_name'], run_name=cfg['run_name'])
        else:
            wandb_logger = WandBLogger(project_name=cfg['project_name'], run_name=cfg['run_name'])
        loggers.append(wandb_logger)


    return EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch_running=True, stream=True),
        loss_metrics(epoch_running=True, stream=True),
        amca_metrics(streams=("test", "train", "val")),
        timing_metrics(epoch_running=True),
        loggers=loggers)
