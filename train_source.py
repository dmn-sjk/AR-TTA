import torch
from torchvision import models
from torch.nn import Linear
from torch.utils.data import DataLoader
import os
import wandb
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from tqdm import tqdm
from robustbench.utils import load_model
from sklearn.metrics import f1_score


from utils.transforms import get_transforms
from utils.config_parser import ConfigParser
from utils.utils import set_seed
from datasets.shift import SHIFTClassificationDataset
from datasets.cifar10c import CIFAR10CDataset
import clad
from shift_dev.types import WeathersCoarse, TimesOfDayCoarse
from shift_dev.utils.backend import ZipBackend


def main():
    cfg = ConfigParser(mode="source").get_config()

    cfg['device'] = torch.device(f"cuda:{cfg['cuda']}"
                                 if torch.cuda.is_available() and cfg['cuda'] >= 0
                                 else "cpu")
    
    if cfg['seed'] is not None:
        set_seed(cfg['seed'])
        
    train_transform = get_transforms(cfg, train=True)
    val_transform = get_transforms(cfg, train=False)

    if cfg['dataset'] == 'shift':
        train_set = SHIFTClassificationDataset(split='train', data_root=cfg['data_root'], transforms=train_transform,
                                                weathers_coarse=[WeathersCoarse.clear], timeofdays_coarse=[TimesOfDayCoarse.daytime],
                                                backend=ZipBackend(), classification_img_size=cfg['img_size'])
        val_set = SHIFTClassificationDataset(split='val', data_root=cfg['data_root'], transforms=val_transform,
                                            weathers_coarse=[WeathersCoarse.clear], timeofdays_coarse=[TimesOfDayCoarse.daytime],
                                            backend=ZipBackend(), classification_img_size=cfg['img_size'])
    elif cfg['dataset'] == 'cifar10c':
        train_set = CIFAR10CDataset(cfg['data_root'], corruption=None, split='train', transforms=train_transform)
        val_set = CIFAR10CDataset(cfg['data_root'], corruption=None, split='test', transforms=val_transform)
    elif cfg['dataset'] == 'clad':
        train_set = clad.get_cladc_train(cfg['data_root'], transform=train_transform, img_size=cfg['img_size'], sequence_type='source')[0]
        # TODO: for now val set has all the domains, maybe modify for only daytime and depending on the possibilities match the weather with train set 
        val_set = clad.get_cladc_val(cfg['data_root'], transform=val_transform, img_size=cfg['img_size'])
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")
    
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)

    if cfg['model'] == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif cfg['model'] == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif cfg['model'] == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif cfg['model'] == 'wideresnet28':
        model = load_model('Standard', cfg['model_ckpt_dir'],
                            'cifar10', "corruptions")
    else:
        raise ValueError(f"Unknown model name: {cfg['model']}")

    model.fc = Linear(model.fc.in_features, cfg['num_classes'], bias=True)
    
    if 'pretrained_model_path' in cfg.keys():
        model.load_state_dict(torch.load(cfg['pretrained_model_path']))

    model.to(cfg['device'])

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=cfg['scheduler_gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    models_path = "models_checkpoints"
    if not os.path.isdir(models_path):
        os.makedirs(models_path)

    if cfg['wandb']:
        wandb.init(config=cfg, project=cfg['project_name'], group=cfg['dataset'],
                   name=f"{cfg['model']}_{cfg['run_name']}", job_type=f"{cfg['model']}_{cfg['run_name']}")

    EVAL_EVERY_EPOCHS = 1

    best_loss = 1e6
    for epoch in range(cfg['num_epochs']):
        model.train()
        last_lr = scheduler.get_last_lr()[-1]

        if cfg['wandb']:
            wandb.log({"lr": last_lr}, step=epoch * len(train_loader))

        with tqdm(enumerate(train_loader), unit="batch", total=len(train_loader)) as tepoch:
            for i, data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, targets = data
                inputs, targets = inputs.to(cfg['device']), targets.to(cfg['device'])

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(outputs, dim=-1)

                acc = (preds == targets).float().mean() * 100.0
                f1 = f1_score(targets, preds, average='macro')
                if cfg['wandb']:
                    wandb.log({'train_accuracy': acc.item(), 'train_loss': loss.item(), 'train_f1': f1.item()},
                              step=epoch*len(train_loader) + i)

                tepoch.set_postfix(loss=loss.item(), accuracy=acc)

        scheduler.step()

        # EVAL
        if epoch % EVAL_EVERY_EPOCHS == EVAL_EVERY_EPOCHS - 1:
            model.eval()
            vloss_sum = 0
            vacc_sum = 0
            vf1_sum = 0
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader, desc="Validating:")):
                    inputs, targets = data
                    inputs, targets = inputs.to(cfg['device']), targets.to(cfg['device'])
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    vloss_sum += loss.item()

                    preds = torch.argmax(outputs, dim=-1)

                    acc = (preds == targets).float().mean() * 100.0
                    vacc_sum += acc.item()
                    
                    vf1 = f1_score(targets, preds, average='macro')
                    vf1_sum += vf1.item()

                avg_vloss = vloss_sum / len(val_loader)
                avg_vacc = vacc_sum / len(val_loader)
                avg_vf1 = vf1_sum / len(val_loader)

                print(
                    f"Val accuracy: {avg_vacc:.2f}\nVal F1: {avg_vf1:.2f}\nVal loss: {avg_vloss:.4f}")

                if cfg['wandb']:
                    wandb.log({'val_accuracy': avg_vacc, 'val_loss': avg_vloss, 'val_f1': avg_vloss},
                              step=(epoch + 1) * len(train_loader))

                if avg_vloss < best_loss:
                    torch.save(model.state_dict(),
                               os.path.join(models_path, f"{cfg['dataset']}_{cfg['model']}_{cfg['run_name']}.pth"))
                    best_loss = avg_vloss

    if cfg['wandb']:
        wandb.finish()


if __name__ == '__main__':
    main()
