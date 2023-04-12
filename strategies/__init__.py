from avalanche.training import Naive
import torchvision
import torch
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, amca_metrics
from avalanche.logging import TextLogger, InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive

from .frozen_strategy import FrozenModel
from loggers.improved_wandb_logger import ImprovedWandBLogger


def get_strategy(cfg):
    if cfg['model'] == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif cfg['model'] == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model name: {cfg['model']}")

    model.fc = torch.nn.Linear(model.fc.in_features, 6, bias=True)

    if cfg['pretrained_model_path'] is not None:
        model.load_state_dict(torch.load(cfg['pretrained_model_path']))

    loggers = [InteractiveLogger()]
    if cfg['text_logger']:
        loggers.append(TextLogger(open(f"./experiments/{cfg['run_name']}/{cfg['run_name']}.log", 'w')))
    if cfg['wandb']:
        if cfg['watch_model']:
            wandb_logger = ImprovedWandBLogger(model=model, project_name=cfg['project_name'], run_name=cfg['run_name'])
        else:
            wandb_logger = WandBLogger(project_name=cfg['project_name'], run_name=cfg['run_name'])
        loggers.append(wandb_logger)

    plugins = []
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch_running=True, stream=True),
        loss_metrics(epoch_running=True, stream=True),
        amca_metrics(streams=("test", "train", "val_sets")),
        loggers=loggers)
        
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 10

    tented_model = None

    if cfg['method'] == "finetune":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        strategy = Naive(
            model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=128,
            device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    elif cfg['method'] == "frozen":
        strategy = FrozenModel(
            model, train_mb_size=batch_size, eval_mb_size=128,
            device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    elif cfg['method'] == "tent":
        pass
        # model, params = get_tented_model_and_params(model)
        # optimizer = torch.optim.SGD(params, lr=1e-3)
        #
        # from tent import softmax_entropy
        #
        # def softmax_entropy_loss(x: torch.Tensor, _):
        #     return softmax_entropy(x).mean(0)
        #
        # criterion = softmax_entropy_loss
        #
        # plugins.append(TentPlugin())
        #
        # strategy = Naive(
        #     model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=128,
        #     device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)

        # plugins.append(TentPlugin(lr=1e-3))

        # ----

        # import tent
        # model = tent.configure_model(model)
        # params, param_names = tent.collect_params(model)
        # optimizer = torch.optim.SGD(params, lr=1e-3)
        # tented_model = tent.Tent(model, optimizer)

        # strategy = FrozenModel(
        #     tented_model, train_mb_size=batch_size, eval_mb_size=32,
        #     device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    else:
        raise ValueError("Unknown method")
    
    return strategy