from typing import List, TYPE_CHECKING, TextIO

import torch
import os
import json
import sys

from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import BaseLogger
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import EpochClassAccuracy
from avalanche.evaluation.metrics import Mean
from collections import defaultdict
from typing import Dict, Set

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class JSONLogger(BaseLogger, SupervisedPlugin):
    """JSONLogger.

    The `JSONLogger` logs accuracy and loss metrics into a json file.

    This logger will log training accuracy for each batch 
    and average accuracy for whole stream of evaluation. 
    
    It does not take into account task ids. It will just append the accuracies
    to the result arrays of the specific stream type 
    (single array for each stream type, e.g., train, test, 
    and possibly more custom streams).
    
    Validation results during training will not be logged.
    """

    # TODO: figure out more reasonable way to save this
    # (results dict is kept in RAM and the whole log file gets updated,
    # since there is no easy way to append values to the array inside json)

    def __init__(self, num_classes: int, log_file: TextIO = sys.stdout):
        """Creates an instance of `JSONLogger` class.

        :param log_file_path: path to results json file, including file name.
        """

        super().__init__()
        self.log_file = log_file
        self.results_dict = {}
        self.training_task_counter = 0
        self.num_classes = num_classes

        self.per_class_predictions = torch.zeros(size=(num_classes,))
        self.per_class_samples = torch.zeros(size=(num_classes,))
        
    def _update_json_file(self):
        self.log_file.seek(0)  # rewind
        json.dump(self.results_dict, self.log_file)
        self.log_file.truncate()

    def _append_results(self, result_key: str, results):
        results = self._prepare_val(results)
        if result_key in self.results_dict.keys():
            self.results_dict[result_key].append(results)
        else:
            self.results_dict[result_key] = [results]

    @staticmethod
    def _prepare_val(m_val):
        if isinstance(m_val, torch.Tensor):
            return m_val.item()
        else:
            return m_val
        
    def after_training_iteration(self, strategy: "SupervisedTemplate", 
                                 metric_values: List["MetricValue"], 
                                 **kwargs):
        super().after_training_iteration(strategy, metric_values, **kwargs)

        # current models outputs
        outputs = strategy.mb_output
        preds = torch.argmax(outputs, dim=-1)
        self.per_class_predictions += torch.bincount(preds.detach().cpu(), minlength=self.num_classes)
        self.per_class_samples += torch.bincount(strategy.mb_y.detach().cpu(), minlength=self.num_classes)

    def after_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().after_training(strategy, metric_values, **kwargs)

        self._append_results('per_class_predictions', self.per_class_predictions.tolist())
        self._append_results('per_class_samples', self.per_class_samples.tolist())
        
        self.per_class_predictions = torch.zeros(size=(self.num_classes,))
        self.per_class_samples = torch.zeros(size=(self.num_classes,))

        classes_not_logged_acc = list(range(self.num_classes))

        # gotta do this because batch-wise acc is not passed via metric_values
        # (generally no metrics are passed after training for some reason)
        metric_values = strategy.evaluator.get_all_metrics()
        keys_to_remove = []
        for key, val in metric_values.items():
            if "train_phase" in key:
                
                if key.startswith("Top1_Acc_MB"):
                    # result_key = <metric_type>/<phase (train/eval)>/<stream_name>
                    # without task id
                    result_key =  key.rsplit('/', 1)[0]
                    self._append_results(result_key, val[1])
                    strategy.evaluator.all_metric_results[key] = [[], []]
                elif key.startswith("Time_Epoch"):
                    result_key = key.split('/')[:-1]
                    result_key[0] = 'AvgTimeIter'

                    new_result_key = ''
                    for i, part in enumerate(result_key):
                        if i != 0:
                            new_result_key += '/'
                        new_result_key += part 

                    avg_iteration_time = val[1][0] / val[0][0]
                    self._append_results(new_result_key, avg_iteration_time)
                    strategy.evaluator.all_metric_results[key] = [[], []]
                elif key.startswith("Top1_ClassAcc_Epoch"):
                    keys_to_remove.append(key)
                    # delete task id
                    key_splitted = key.split('/')
                    del key_splitted[3]
                    result_key = '/'.join(key_splitted)
                    self._append_results(result_key, val[1][0])
                    
                    class_id = key_splitted[-1]
                    classes_not_logged_acc.remove(int(class_id))
                    
        for class_id in classes_not_logged_acc:
            result_key = f"Top1_ClassAcc_Epoch/train_phase/train_stream/{class_id}"
            self._append_results(result_key, None)

        # for saving per class acc
        for key in keys_to_remove:
            del strategy.evaluator.all_metric_results[key]

        for i, metric in enumerate(strategy.evaluator.metrics):
            if isinstance(metric, EpochClassAccuracy):
                strategy.evaluator.metrics[i]._class_accuracy._class_accuracies = defaultdict(lambda: defaultdict(Mean))
                strategy.evaluator.metrics[i]._class_accuracy.classes: Dict[int, Set[int]] = defaultdict(set)

        self._update_json_file()
        self.training_task_counter += 1

    def after_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().after_eval(strategy, metric_values, **kwargs)

        for val in metric_values:
            if "eval_phase" in val.name:
                if val.name.startswith("Top1_Acc_Stream"):
                    # result_key = <metric_type>/<phase (train/eval)>/<stream_name>
                    # without task id
                    result_key =  val.name.rsplit('/', 1)[0]
                    self._append_results(result_key, val.value)
                    
        self._update_json_file()

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().before_training(strategy, metric_values, **kwargs)

    def close(self):
        self.log_file.close()