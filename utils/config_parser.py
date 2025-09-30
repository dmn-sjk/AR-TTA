import argparse
import os
import sys
from typing import Dict, Set

import yaml

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
        self.explicit_args: Set[str] = set()  # Track explicitly provided arguments
        
    def get_config(self) -> Dict:
        args = self._parse_args()
        cfg = self._read_config(args)
        cfg = self._overwrite_config_with_args(cfg, args)
        print(cfg)
        return cfg
    
    def get_explicit_args(self) -> Set[str]:
        """Return set of argument names that were explicitly provided via command line."""
        return self.explicit_args.copy()
    
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
    
    def _setup_parser(self) -> argparse.ArgumentParser:
        """Set up the argument parser with all arguments."""
        parser = argparse.ArgumentParser(
            description='Code for TTA testing. \
                \nSome params are already defined in .yaml files in configs folder. \
                    Command line arguments overwrite params defined in .yaml configs')
        
        parser.add_argument('--dataset', type=str, default=None, required=True)
        parser.add_argument('--data_root', type=str, default=None, required=True,
                            help='Root folder where the data is stored')
        parser.add_argument('--model', type=str, default=None, required=True,
                            help='Model architecture')
        parser.add_argument('--run_name', type=str, default='',
                            help='Name of the run')
        if self.mode == 'tta':
            valid_methods = _METHODS.keys()
            parser.add_argument('--method', type=str, choices=valid_methods, default=None, required=True,
                                help=' | '.join(valid_methods))
        parser.add_argument('--src_model_ckpt_file', type=str, default=None,
                            help='Name of the source model ckpt file')
        parser.add_argument('--ckpts_dir', type=str, default='models_checkpoints',
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

        # optimizer
        parser.add_argument('--optimizer', type=str, default='sgd',
                            help='Optimizer')
        parser.add_argument('--scheduler_gamma', type=float, default=0.85,
                            help="Gamma value for exponential lr scheduler")
        parser.add_argument('--lr', type=float, default=0.01,
                            help="Learning rate")
        parser.add_argument('--nesterov', action='store_true',
                            help="Use Nesterov momentum")
        parser.add_argument('--weight_decay', type=float, default=0.0,
                            help="Weight decay for optimizer")
        parser.add_argument('--beta', type=float, default=0.9,
                            help="Beta1 parameter for Adam optimizer")

        # AR-TTA
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
        
        # CoTTA
        parser.add_argument('--mt', type=float, default=0.999)
        parser.add_argument('--rst', type=float, default=0.01)
        parser.add_argument('--ap', type=float, default=0.92)
        
        # EATA/SAR
        parser.add_argument('--fisher_alpha', type=float, default=1.0)
        parser.add_argument('--fisher_size', type=int, default=2000)
        parser.add_argument('--d_margin', type=float, default=0.4)
        parser.add_argument('--e_margin_coeff', type=float, default=0.4)
        
        return parser

    def _parse_args(self) -> None:
        # First, track which arguments were explicitly provided by parsing sys.argv
        self._track_explicit_args()
        
        # Use the same parser setup
        parser = self._setup_parser()
        return parser.parse_args()

    def _track_explicit_args(self) -> None:
        """Track which arguments were explicitly provided by parsing sys.argv."""
        # Create a temporary parser to get all argument names
        temp_parser = self._setup_parser()
        
        # Extract argument names from the parser
        all_arg_names = set()
        for action in temp_parser._actions:
            if hasattr(action, 'dest') and action.dest != 'help':
                all_arg_names.add(action.dest)
        
        # Check which arguments appear in sys.argv
        for i, arg in enumerate(sys.argv[1:], 1):  # Skip script name
            if arg.startswith('--'):
                arg_name = arg[2:]  # Remove '--' prefix
                if arg_name in all_arg_names:
                    self.explicit_args.add(arg_name)

    def _overwrite_config_with_args(self, config: Dict, args: argparse.Namespace) -> Dict:
        # explicit args > configs > default args
        _args = vars(args)
        for key, value in config.items():
            if key in self.explicit_args:
                continue
            _args[key] = value

        # 'None' to None
        for key, val in _args.items():
            if val == 'None':
                _args[key] = None

        return _args

