import yaml
import os
import argparse
from typing import Dict

from methods import _METHODS


class ConfigParser:
    """
    Class for parsing config and arguments.

    Command-line arguments have priority over yaml config values.
    """

    GENERAL_CONFIG_FILE = "general"
    CONFIGS_DIR = "configs"
    
    def __init__(self, mode: str) -> None:
        """Parse 

        Args:
            mode (str): source | tta
        """        
        if mode != 'source' and mode != 'tta':
            raise ValueError(f"Unknown mode: {mode}") 
        
        self.mode = mode
        
    def get_config(self) -> Dict:
        args = self._parse_args()
        cfg = self._read_config(args)
        cfg = self._overwrite_config_with_args(cfg, args)
        print(cfg)
        return cfg
    
    def _read_config(self, args: argparse.Namespace) -> None:
        if self.mode == 'source':
            cfg_files = [
                os.path.join(self.CONFIGS_DIR, "datasets", args.dataset + ".yaml")
            ]
        else:
            cfg_files = [
                os.path.join(self.CONFIGS_DIR, "methods", args.dataset, args.method + ".yaml"),
                os.path.join(self.CONFIGS_DIR, "datasets", args.dataset + ".yaml")
            ]
        
        cfg = {}
        for cfg_file in cfg_files:
            with open(cfg_file, "r") as yamlfile:
                file_dict = yaml.safe_load(yamlfile)
                if file_dict is not None:
                    cfg.update(file_dict)
        return cfg
    
    def _parse_args(self) -> None:
        parser = argparse.ArgumentParser(
            description='Code for TTA testing. \nSome params are already defined in .yaml files in configs folder. Command line arguments overwrite params defined in .yaml configs')
        parser.add_argument('--dataset', type=str, default=None, required=True)
        parser.add_argument('--run_name', type=str, default=None, required=True,
                            help='Name of the run')
        parser.add_argument('--data_root', type=str, default=None, required=True,
                            help='Root folder where the data is stored')
        parser.add_argument('--model', type=str, default=None, required=True,
                            help='Model architecture')
        if self.mode == 'tta':
            valid_methods = _METHODS.keys()
            parser.add_argument('--method', type=str, choices=valid_methods, default=None, required=True,
                                help=' | '.join(valid_methods))
        parser.add_argument('--pretrained_model_path', type=str, default=None,
                            help='path to pretrained model')
        parser.add_argument('--model_ckpt_dir', type=str, default='models_checkpoints',
                            help='Directory of model checkpoints')
        parser.add_argument('--log_dir', type=str, default='logs',
                            help='General log directory')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Num workers to use for dataloading')
        parser.add_argument('--cuda', type=int, default=0,
                            help='Whether to use cuda and which GPU to use, -1 if not')
        parser.add_argument('--seed', type=int, default=1234,
                            help='Random seed')
        parser.add_argument('--batch_size', type=int, default=10,
                            help="Batch size")
        parser.add_argument('--num_epochs', type=int, default=-1,
                            help="The number of epochs for training. -1 for unlimited epochs")
        parser.add_argument('--img_size', type=int, default=224,
                            help="Size of images to use")
        parser.add_argument('--scheduler_gamma', type=float, default=0.85,
                            help="Gamma value for exponential lr scheduler")
        parser.add_argument('--lr', type=float, default=0.01,
                            help="Learning rate")
        parser.add_argument('--init_beta', type=float, default=0.1,
                            help="Beta for BN stats ema")
        parser.add_argument('--bn_dist_scale', type=float, default=10,
                            help="Scale for distributions distance in dynamic BN")
        parser.add_argument('--alpha', type=float, default=0.4,
                            help="Param of beta distribution for mixup")
        parser.add_argument('--smoothing_beta', type=float, default=0.2,
                            help="Coeff for ema beta")
        parser.add_argument('--memory_size', type=int, default=2000,
                            help="Size of class-balanced memory")
        parser.add_argument('--save_results', action='store_true',
                            help="Save results")
        return parser.parse_args()

    def _overwrite_config_with_args(self, config: Dict, args: argparse.Namespace) -> Dict:
        config.update(vars(args))
        return config

