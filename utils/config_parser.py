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
                os.path.join(self.CONFIGS_DIR, "methods", args.method + ".yaml"),
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
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default=None, required=True,
                            help='clad | shift | cifar10c')
        parser.add_argument('--run_name', type=str, default=None, required=True,
                            help='Name of the run')
        if self.mode == 'tta':
            parser.add_argument('--benchmark', type=str, default=None, required=True,
                                help='cifar10c_long | cifar10c_repetitive | cifar10c_standard | clad | shift_weather | shift_timeofday | shift_mix')
            parser.add_argument('--method', type=str, default=None, required=True,
                                help='source | finetune | tent | cotta | eata')
        parser.add_argument('--model', type=str, default=argparse.SUPPRESS,
                            help='Name of the run')
        parser.add_argument('--project_name', type=str, default=argparse.SUPPRESS,
                            help='Name of the run')
        parser.add_argument('--pretrained_model_path', type=str, default=argparse.SUPPRESS,
                            help='Name of the run')
        parser.add_argument('--data_root', type=str, default=argparse.SUPPRESS,
                            help='Root folder where the data is stored')
        parser.add_argument('--note', type=str, default=argparse.SUPPRESS,
                            help='Notes for the run')
        parser.add_argument('--num_workers', type=int, default=argparse.SUPPRESS,
                            help='Num workers to use for dataloading')
        parser.add_argument('--cuda', type=int, default=argparse.SUPPRESS,
                            help='Whether to use cuda, -1 if not')
        parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                            help='Random seed')
        parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                            help="Log to .txt")
        parser.add_argument('--num_epochs', type=int, default=argparse.SUPPRESS,
                            help="Log to .txt")
        parser.add_argument('--img_size', type=int, default=argparse.SUPPRESS,
                            help="Log to .txt")
        parser.add_argument('--scheduler_gamma', type=float, default=argparse.SUPPRESS,
                            help="Log to .txt")
        parser.add_argument('--distillation_temp', type=int, default=argparse.SUPPRESS,
                            help="Log to .txt")
        parser.add_argument('--wandb', action='store_true',
                            help="Log with wandb")
        parser.add_argument('--save_results', action='store_true',
                            help="Log to .txt")
        parser.add_argument('--watch_model', action='store_true',
                            help="Log model state in wandb")
        return parser.parse_args()

    def _overwrite_config_with_args(self, config: Dict, args: argparse.Namespace) -> Dict:
        config.update(vars(args))
        return config
        
        # for key, val in config.items():
        #     if isinstance(val, dict):
        #         self._overwrite_config_with_args(val)
        #     else:
        #         if key in vars(self.args).keys():
        #             arg_val = getattr(self.args, key)
        #             if arg_val is not None:
        #                 config[key] = arg_val
