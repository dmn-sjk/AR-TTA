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
        self.mode = mode
        
    def get_config(self) -> Dict:
        args = self._parse_args()
        cfg = self._read_config(args)
        cfg = self._overwrite_config_with_args(cfg, args)
        return cfg
    
    def _read_config(self, args: argparse.Namespace) -> None:
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
        parser.add_argument('--method', type=str, default=None, required=True,
                            help='source | finetune | tent | cotta | note | adacon')
        parser.add_argument('--dataset', type=str, default=None, required=True,
                            help='clad | shift | cifar10c')
        if self.mode == 'tta':
            parser.add_argument('--benchmark', type=str, default=None, required=True,
                                help='cifar10c_long | cifar10c_repetitive | clad | shift_weather | shift_timeofday')
        parser.add_argument('--project_name', type=str, default=argparse.SUPPRESS,
                            help='Name of the run')
        parser.add_argument('--run_name', type=str, default=argparse.SUPPRESS,
                            help='Name of the run')
        parser.add_argument('--model', type=str, default=argparse.SUPPRESS,
                            help='Name of the run')
        parser.add_argument('--pretrained_model_path', type=str, default=argparse.SUPPRESS,
                            help='Name of the run')
        parser.add_argument('--data_root', default=argparse.SUPPRESS,
                            help='Root folder where the data is stored')
        parser.add_argument('--num_workers', type=int, default=argparse.SUPPRESS,
                            help='Num workers to use for dataloading')
        parser.add_argument('--cuda', type=int, default=argparse.SUPPRESS,
                            help='Whether to use cuda, -1 if not')
        parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                            help='Random seed')
        parser.add_argument('--store', action='store_true',
                            help="If set the prediciton files required for submission will be created")
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
