import yaml
import os
import argparse
from typing import Dict


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
                os.path.join(self.CONFIGS_DIR, "general", self.GENERAL_CONFIG_FILE + ".yaml"),
                os.path.join(self.CONFIGS_DIR, "general", self.GENERAL_CONFIG_FILE + "_" + self.mode + ".yaml"),
                os.path.join(self.CONFIGS_DIR, "datasets", args.dataset + ".yaml")
            ]
        else:
            cfg_files = [
                os.path.join(self.CONFIGS_DIR, "general", self.GENERAL_CONFIG_FILE + ".yaml"),
                os.path.join(self.CONFIGS_DIR, "general", self.GENERAL_CONFIG_FILE + "_" + self.mode + ".yaml"),
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
        parser.add_argument('--dataset', type=str, default=None, required=True,
                            help='clad | shift | cifar10c')
        parser.add_argument('--run_name', type=str, default=None, required=True,
                            help='Name of the run')
        if self.mode == 'tta':
            parser.add_argument('--method', type=str, default=None, required=True,
                                help='frozen | finetune | tent | cotta | eata | sar | artta | bn_stats_adapt | rmt')
        parser.add_argument('--model', type=str, default=argparse.SUPPRESS,
                            help='name of the model')
        parser.add_argument('--project_name', type=str, default=argparse.SUPPRESS,
                            help='project name for wandb')
        parser.add_argument('--pretrained_model_path', type=str, default=argparse.SUPPRESS,
                            help='path to pretrained model')
        parser.add_argument('--data_root', type=str, default=argparse.SUPPRESS,
                            help='Root folder where the data is stored')
        parser.add_argument('--notes', type=str, default=argparse.SUPPRESS,
                            help='Notes for the run')
        parser.add_argument('--num_workers', type=int, default=argparse.SUPPRESS,
                            help='Num workers to use for dataloading')
        parser.add_argument('--cuda', type=int, default=argparse.SUPPRESS,
                            help='Whether to use cuda and which GPU to use, -1 if not')
        parser.add_argument('--seeds', type=lambda s: [int(item) for item in s.strip().split(',')], default=argparse.SUPPRESS,
                            help='List of random seeds. Use comma to delimeter: --seeds 1234,1235,1236')
        parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                            help="Training batch size")
        parser.add_argument('--num_epochs', type=int, default=argparse.SUPPRESS,
                            help="The number of epochs for training")
        parser.add_argument('--img_size', type=int, default=argparse.SUPPRESS,
                            help="Size of images to use")
        parser.add_argument('--scheduler_gamma', type=float, default=argparse.SUPPRESS,
                            help="Gamma value for exponential lr scheduler")
        parser.add_argument('--lr', type=float, default=argparse.SUPPRESS,
                            help="Learning rate")
        parser.add_argument('--init_beta', type=float, default=argparse.SUPPRESS,
                            help="Beta for stats ema")
        parser.add_argument('--bn_dist_scale', type=float, default=argparse.SUPPRESS,
                            help="Scale for distributions distance in dynamic BN")
        parser.add_argument('--alpha', type=float, default=argparse.SUPPRESS,
                            help="For beta distrib")
        parser.add_argument('--smoothing_beta', type=float, default=argparse.SUPPRESS,
                            help="Coeff for ema beta")
        parser.add_argument('--memory_size', type=int, default=argparse.SUPPRESS,
                            help="Size of class-balanced memory")
        parser.add_argument('--bn_stats', type=str, default=argparse.SUPPRESS,
                            help='source | test | dynamicbn')
        parser.add_argument('--rank_mode', type=str, default=argparse.SUPPRESS,
                            help='fixed | threshold | ratio | percentile')
        parser.add_argument('--wandb', action='store_true',
                            help="Log with wandb")
        parser.add_argument('--universal', type=float, default=argparse.SUPPRESS,
                            help="param for various tests")
        parser.add_argument('--save_results', action='store_true',
                            help="Save results")
        return parser.parse_args()

    def _overwrite_config_with_args(self, config: Dict, args: argparse.Namespace) -> Dict:
        config.update(vars(args))
        return config

