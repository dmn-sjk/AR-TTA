from .tensorboard_logger import TensorBoardLogger
from typing import Optional, Dict, Any


# def get_tensorboard_logger(cfg: Dict[str, Any], 
#                           experiment_name: Optional[str] = None,
#                           seed: Optional[int] = None) -> TensorBoardLogger:
#     """
#     Create a TensorBoard logger based on configuration.
    
#     Args:
#         cfg: Configuration dictionary
#         experiment_name: Override experiment name from config
#         seed: Override seed from config
        
#     Returns:
#         Initialized TensorBoardLogger instance
#     """
#     # Get experiment name
#     if experiment_name is None:
#         if 'experiment_name' in cfg:
#             experiment_name = cfg['experiment_name']
#         else:
#             # Create experiment name from config
#             dataset = cfg.get('dataset', 'unknown')
#             method = cfg.get('method', 'unknown')
#             model = cfg.get('model', 'unknown')
#             run_name = cfg.get('run_name', 'default')
#             experiment_name = f"{dataset}_{method}_{model}_{run_name}"
    
#     # Get seed
#     if seed is None:
#         if 'curr_seed' in cfg:
#             seed = cfg['curr_seed']
#         elif 'seeds' in cfg and len(cfg['seeds']) > 0:
#             seed = cfg['seeds'][0]
    
#     # Get log directory
#     log_dir = cfg.get('log_dir', 'logs')
#     tensorboard_log_dir = f"{log_dir}/tensorboard"
    
#     return TensorBoardLogger(
#         log_dir=tensorboard_log_dir,
#         experiment_name=experiment_name,
#         seed=seed
#     )


# def log_config_to_tensorboard(logger: TensorBoardLogger, cfg: Dict[str, Any]):
#     """
#     Log configuration parameters to TensorBoard.
    
#     Args:
#         logger: TensorBoardLogger instance
#         cfg: Configuration dictionary
#     """
#     # Convert config to flat dictionary for logging
#     flat_config = {}
    
#     def flatten_dict(d, parent_key='', sep='_'):
#         items = []
#         for k, v in d.items():
#             new_key = f"{parent_key}{sep}{k}" if parent_key else k
#             if isinstance(v, dict):
#                 items.extend(flatten_dict(v, new_key, sep=sep).items())
#             else:
#                 items.append((new_key, v))
#         return dict(items)
    
#     flat_config = flatten_dict(cfg)
    
#     # Log hyperparameters (will be logged when first metrics are added)
#     logger.log_hparams(flat_config, {})


# __all__ = ['TensorBoardLogger']


