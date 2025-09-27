import torch

from datasets import get_num_classes

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
                
        self.num_correct = 0
        self.num_samples = 0
        self.num_correct_per_class = torch.zeros((self.cfg['num_classes'],), dtype=torch.int)
        self.num_samples_per_class = torch.zeros((self.cfg['num_classes'],), dtype=torch.int)

    def add_preds(self, preds: torch.Tensor, labels: torch.Tensor):
        _num_samples = len(labels)
        _num_correct = (preds == labels).int().sum()

        _num_samples_per_class = torch.bincount(labels, minlength=self.cfg['num_classes'])
        _num_correct_per_class = torch.bincount(labels[preds == labels], minlength=self.cfg['num_classes'])

        acc = (_num_correct / _num_samples) * 100.0
        # nan for classes which are not in the batch
        per_class_acc = (_num_correct_per_class / _num_samples_per_class) * 100.0

        self.num_correct += _num_correct
        self.num_samples += _num_samples
        self.num_correct_per_class += _num_correct_per_class
        self.num_samples_per_class += _num_samples_per_class
        
        mca = per_class_acc.nanmean()
        return acc, mca, per_class_acc
    
    def get_summary(self):
        overall_acc = (self.num_correct / self.num_samples) * 100.0
        overall_acc_per_class = (self.num_correct_per_class / self.num_samples_per_class) * 100.0
        mca = overall_acc_per_class.nanmean()
        return overall_acc, mca, overall_acc_per_class, \
            self.num_samples, self.num_correct, self.num_samples_per_class, self.num_correct_per_class

    def reset(self):
        self.num_correct = 0
        self.num_samples = 0
        self.num_correct_per_class = [0 for _ in range(self.cfg['num_classes'])]
        self.num_samples_per_class = [0 for _ in range(self.cfg['num_classes'])]