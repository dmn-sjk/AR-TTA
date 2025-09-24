from typing import Optional, Sequence, List, Union

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate, BaseTemplate
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates.base import _group_experiences_by_stream
from avalanche.benchmarks import CLExperience, CLStream
from typing import Iterable, Sequence, Optional, Union, List
from avalanche.training.templates.observation_type import OnlineObservation
from avalanche.models.dynamic_modules import MultiTaskModule

ExpSequence = Iterable[CLExperience]

import torch

from avalanche.training.templates import SupervisedTemplate, BaseTemplate
from avalanche.core import SupervisedPlugin
from . import register_strategy


@register_strategy("frozen")
def get_frozen_strategy(cfg, model: Module, eval_plugin: EvaluationPlugin, plugins: Sequence):
    return FrozenModel(model, train_mb_size=cfg['batch_size'], eval_mb_size=128,
                       device=cfg['device'], evaluator=eval_plugin, plugins=plugins, eval_every=-1)


class FrozenPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.eval()
        for param in strategy.model.parameters():
            param.requires_grad = False


class FrozenModel(SupervisedTemplate):
    """Naive finetuning.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(
        self,
        model: Module,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        # for param in model.parameters():
        #     param.requires_grad = False

        # optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
        dummy_optimizer = SGD(model.parameters(), lr=0.01)

        super().__init__(
            model,
            optimizer=dummy_optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=1,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

    @torch.no_grad()
    def train(
            self,
            experiences: Union[CLExperience, ExpSequence],
            eval_streams: Optional[
                Sequence[Union[CLExperience, ExpSequence]]
            ] = None,
            **kwargs,
    ):
        """Training loop.

        If experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: sequence of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.
            Experiences in `eval_streams` are grouped by stream name
            when calling `eval`. If you use multiple streams, they must
            have different names.
        """
        self.is_training = True
        self._stop_training = False

        # self.model.train()
        self.model.eval()
        self.model.to(self.device)

        # Normalize training and eval data.
        if not isinstance(experiences, Iterable):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]

        self._eval_streams = _group_experiences_by_stream(eval_streams)

        self._before_training(**kwargs)

        for self.experience in experiences:
            self._before_training_exp(**kwargs)
            self._train_exp(self.experience, eval_streams, **kwargs)
            self._after_training_exp(**kwargs)
        self._after_training(**kwargs)

    @torch.no_grad()
    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            # self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            # self._before_backward(**kwargs)
            # # self.backward()
            # self._after_backward(**kwargs)
            #
            # # Optimization step
            # self._before_update(**kwargs)
            # # self.optimizer_step()
            # self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

# UNCOMMENT BELOW FOR GT LABELS IN BATCHES
#     def forward(self):
#         """Compute the model's output given the current mini-batch."""
#         return avalanche_forward(self.model, self.mbatch, self.mb_task_id)
    
# def avalanche_forward(model, x, task_labels):
#     if isinstance(model, MultiTaskModule):
#         return model(x, task_labels)
#     else:  # no task labels
#         return model(x)
