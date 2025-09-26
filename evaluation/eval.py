import torch

from .tensorboard_logger import TensorBoardLogger
from .evaluator import Evaluator


def eval_domain(cfg, model, dataloader: torch.utils.data.DataLoader, logger: TensorBoardLogger):
    evaluator = Evaluator(cfg)
    
    for i, (x, y) in enumerate(dataloader):
        x = x.to(cfg['device'])
        
        preds = model(x)
        preds = preds.argmax(1).detach().cpu()

        acc, acc_per_class = evaluator.add_preds(preds, y)

        log_dict = {f'acc_class_{i}': acc_per_class[i].item() for i in range(cfg['num_classes'])}
        log_dict['acc'] = acc.item()
        logger.log_scalars('per_batch', log_dict)
        
        if i == len(dataloader) - 1:
            overall_acc, overall_acc_per_class = evaluator.get_summary()
            log_dict = {f'acc_class_{i}': overall_acc_per_class[i].item() for i in range(cfg['num_classes'])}
            log_dict['acc'] = overall_acc.item()
            logger.log_scalars('per_domain',log_dict)
        
        logger.step += 1
        
    return overall_acc, overall_acc_per_class