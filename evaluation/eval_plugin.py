from ..logging.loggers import get_loggers
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, amca_metrics
from avalanche.training.plugins import EvaluationPlugin


def get_eval_plugin(args):
    return None

    loggers = get_loggers(args)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch_running=True, stream=True),
        loss_metrics(epoch_running=True, stream=True),
        amca_metrics(streams=("test", "train", "val_sets")),
        loggers=loggers)