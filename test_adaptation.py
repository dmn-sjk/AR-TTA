from clad import *

import torch
import torchvision.models
from torch.nn import Linear

import argparse

from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, amca_metrics
from avalanche.logging import TextLogger, InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive
import json
from custom_strategies.frozen_strategy import FrozenModel
from custom_strategies.tent_strategy import TentPlugin, TentPluginBetter, get_tented_model_and_params
from torchvision.transforms import transforms
from torch import nn
from custom_loggers.improved_wandb_logger import ImprovedWandBLogger

# TODO: copied, rewrite correctly

def norm_params_unchanged(strategy, prev_norm_params):
        for m in strategy.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                if prev_norm_params['running_mean'] is not None:
                    if not torch.all(torch.eq(prev_norm_params['running_mean'], m.running_mean.cpu())):
                        print('running_mean NOT THE SAME')

                if prev_norm_params['running_var'] is not None:
                    if not torch.all(torch.eq(prev_norm_params['running_var'], m.running_var.cpu())):
                        print('running_var NOT THE SAME')

                prev_norm_params['running_mean'] = m.running_mean.cpu()
                prev_norm_params['running_var'] = m.running_var.cpu()

                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        if prev_norm_params['named'] is not None:
                            if not torch.all(torch.eq(prev_norm_params['named'], p.data.cpu())):
                                print('named NOT THE SAME')

                        prev_norm_params['named'] = p.data.cpu()
                        break
                break

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_parser_and_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='CLAD-C',
                        help='Name of the run')
    parser.add_argument('--run_name', type=str, default='clad-c',
                        help='Name of the run')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Name of the run')
    parser.add_argument('--method', type=str, default='finetune',
                        help='Name of the run')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='Name of the run')
    parser.add_argument('--root', default="/home/damian/Documents/datasets/clad/",
                        help='Root folder where the data is stored')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Num workers to use for dataloading')
    parser.add_argument('--cuda', type=int, default=-1,
                        help='Whether to use cuda, -1 if not')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--store', action='store_true',
                        help="If set the prediciton files required for submission will be created")
    parser.add_argument('--no_cuda', action='store_true',
                        help='If set, training will be on the CPU')
    parser.add_argument('--store_model', action='store_true',
                        help="Stores model if specified. Has no effect is store is not set")
    parser.add_argument('--wandb', action='store_true',
                        help="Log with wandb")
    parser.add_argument('--text_logger', action='store_true',
                        help="Log to .txt")
    parser.add_argument('--save_results', action='store_true',
                        help="Log to .txt")
    parser.add_argument('--watch_model', action='store_true',
                        help="Log to .txt")
    return parser.parse_args()


def main():
    args = setup_parser_and_get_args()

    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and args.cuda >= 0
                          else "cpu"
                          )

    if args.seed is not None:
        set_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    cladc = cladc_avalanche(args.root, test_transform=transform, train_trasform=transform)


    if args.model == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model name: {args.model}")

    model.fc = Linear(model.fc.in_features, len(SODA_CATEGORIES), bias=True)

    if args.pretrained_model_path is not None:
        model.load_state_dict(torch.load(args.pretrained_model_path))


    loggers = [InteractiveLogger()]
    if args.text_logger:
        loggers.append(TextLogger(open(f"./experiments/{args.run_name}/{args.run_name}.log", 'w')))
    if args.wandb:
        if args.watch_model:
            wandb_logger = ImprovedWandBLogger(model=model, project_name=args.project_name, run_name=args.run_name)
        else:
            wandb_logger = WandBLogger(project_name=args.project_name, run_name=args.run_name)
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

    if args.method == "finetune":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        strategy = Naive(
            model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=128,
            device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    elif args.method == "frozen":
        strategy = FrozenModel(
            model, train_mb_size=batch_size, eval_mb_size=128,
            device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    elif args.method == "tent":
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

        import tent
        model = tent.configure_model(model)
        params, param_names = tent.collect_params(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        tented_model = tent.Tent(model, optimizer)

        strategy = FrozenModel(
            tented_model, train_mb_size=batch_size, eval_mb_size=32,
            device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    else:
        raise ValueError("Unknown method")

    results_dict = {"train_sequences": {"running_acc": [], "batch_acc": []},
                    "val_sequences": {"acc": [], "refined_acc": []}, # refined for AdaCon, not used here
                    "test_sequences": {"acc": [], "refined_acc": []}}

    prev_norm_params = {"running_mean": None, "running_var": None, "named": None}

    for i, experience in enumerate(cladc.train_stream[1:]):
        # strategy.train(experience, eval_streams=[cladc.streams['val_sets']], shuffle=False,
        #                num_workers=args.num_workers)

        if i == 0:
            print("Initial eval...")
            results = strategy.eval(cladc.test_stream, num_workers=args.num_workers)
            print(results)
            strategy.eval(cladc.streams['val_sets'], num_workers=args.num_workers)

            if args.save_results:
                metrics = eval_plugin.get_all_metrics()
                results_dict["val_sequences"]["acc"]. \
                    append(metrics[f"Top1_Acc_Stream/eval_phase/val_sets_stream/Task000"][1])
                results_dict["test_sequences"]["acc"]. \
                    append(metrics[f"Top1_Acc_Stream/eval_phase/test_stream/Task000"][1])

        strategy.train(experience, eval_streams=[], shuffle=False,
                       num_workers=args.num_workers)

        results = strategy.eval(cladc.test_stream, num_workers=args.num_workers)
        print(results)
        strategy.eval(cladc.streams['val_sets'], num_workers=args.num_workers)

        if args.save_results:
            metrics = eval_plugin.get_all_metrics()
            results_dict["train_sequences"]["running_acc"]. \
                append(metrics[f"Top1_RunningAcc_Epoch/train_phase/train_stream/Task00{i + 1}"][1])
            results_dict["train_sequences"]["batch_acc"]. \
                append(metrics[f"Top1_Acc_MB/train_phase/train_stream/Task00{i + 1}"][1])

            results_dict["val_sequences"]["acc"]. \
                append(metrics[f"Top1_Acc_Stream/eval_phase/val_sets_stream/Task000"][1])
            results_dict["test_sequences"]["acc"]. \
                append(metrics[f"Top1_Acc_Stream/eval_phase/test_stream/Task000"][1])

    if args.save_results:
        results_path = os.path.join("experiments", f"{args.run_name}_{args.method}")
        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        with open(os.path.join(results_path, f"{args.run_name}_{args.method}_results.json"), 'w') \
                as file:
            json.dump(results_dict, file)


if __name__ == '__main__':
    main()
