import clad

import random
import numpy as np
import torch
import argparse
from torchvision import models
from torch.nn import Linear
from torch.utils.data import DataLoader
import os
import wandb
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from tqdm import tqdm
from utils.transforms import get_transforms
from benchmarks.shift import _SHIFTClassificationDataset
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse

# TODO: whole file copied, rewrite properly

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
    parser.add_argument('--project_name', type=str, default='Continual adaptation',
                        help='Name of the run')
    parser.add_argument('--run_name', type=str, default='source_training',
                        help='Name of the run')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Name of the run')
    parser.add_argument('--data_root', default="/home/damian/Documents/datasets/",
                        help='Root folder where the data is stored')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Num workers to use for dataloading')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Whether to use cuda, -1 if not')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help="Log with wandb")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Log with wandb")
    parser.add_argument('--epochs', type=int, default=15,
                        help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Random seed')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        help='Name of the run')
    return parser.parse_args()


def main():
    args = setup_parser_and_get_args()

    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and args.cuda >= 0
                          else "cpu"
                          )

    if args.seed is not None:
        set_seed(args.seed)
        
    train_set = _SHIFTClassificationDataset(split='train', data_root=args.data_root, transforms=get_transforms(None, train=True),
                                            weathers_coarse=[WeathersCoarse.clear], timeofdays_coarse=[TimesOfDayCoarse.daytime])
    val_set = _SHIFTClassificationDataset(split='val', data_root=args.data_root, transforms=get_transforms(None, train=False),
                                          weathers_coarse=[WeathersCoarse.clear], timeofdays_coarse=[TimesOfDayCoarse.daytime])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    if args.model == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.model == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif args.model == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model name: {args.model}")

    # model.fc = Linear(model.fc.in_features, len(clad.SODA_CATEGORIES), bias=True)
    model.fc = Linear(model.fc.in_features, len(train_set.classes), bias=True)

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.8)

    if args.loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == "triplet_loss":
        raise NotImplementedError()
        # TODO
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")

    models_path = "models_checkpoints"
    if not os.path.isdir(models_path):
        os.mkdir(models_path)

    if args.wandb:
        wandb.init(config=args, project=args.project_name, group="shift_c",
                   name=f"{args.run_name}_{args.model}", job_type=f"{args.run_name}_{args.model}")

    # train_set = clad.get_cladc_train(args.root, transform=get_transforms(None, train=True))[0]
    # val_set = clad.get_cladc_val(args.root, transform=get_transforms(None, train=False))

    EVAL_EVERY_EPOCHS = 1

    best_loss = 1e6
    for epoch in range(args.epochs):
        # loss_sum = 0
        # acc_sum = 0
        model.train()
        last_lr = scheduler.get_last_lr()[-1]

        if args.wandb:
            wandb.log({"lr": last_lr}, step=epoch * len(train_loader))

        with tqdm(enumerate(train_loader), unit="batch", total=len(train_loader)) as tepoch:
            for i, data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                acc = (torch.argmax(outputs, dim=-1) ==
                       targets).float().mean().item() * 100.0
                if args.wandb:
                    wandb.log({'train_accuracy': acc, 'train_loss': loss.item()},
                              step=epoch*len(train_loader) + i)

                tepoch.set_postfix(loss=loss.item(), accuracy=acc)

        scheduler.step()

        # EVAL
        if epoch % EVAL_EVERY_EPOCHS == EVAL_EVERY_EPOCHS - 1:
            model.eval()
            vloss_sum = 0
            vacc_sum = 0
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader, desc="Validating:")):
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    vloss_sum += loss.item()
                    acc = (torch.argmax(outputs, dim=-1)
                           == targets).float().mean()
                    vacc_sum += acc.item()

                avg_vloss = vloss_sum / len(val_loader)
                avg_vacc = vacc_sum / len(val_loader) * 100.0

                print(
                    f"Val accuracy: {avg_vacc:.2f}\nVal loss: {avg_vloss:.4f}")

                if args.wandb:
                    wandb.log({'val_accuracy': avg_vacc, 'val_loss': avg_vloss},
                              step=(epoch + 1) * len(train_loader))

                if avg_vloss < best_loss:
                    torch.save(model.state_dict(),
                               os.path.join(models_path, f"shift_c_{args.model}.pth"))
                    best_loss = avg_vloss

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
