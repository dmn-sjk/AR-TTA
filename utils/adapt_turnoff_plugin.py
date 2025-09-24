from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


# TODO: make sure it is actually used in main experiments, remove if not
class AdaptTurnoffPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        if not hasattr(strategy.model, "adapt"):
            raise AttributeError("AdaptTurnoff plugin requires adapt attribute")

        strategy.model.adapt = True

    def before_eval(self, strategy: "SupervisedTemplate", **kwargs):
        if not hasattr(strategy.model, "adapt"):
            raise AttributeError("AdaptTurnoff plugin requires adapt attribute")

        strategy.model.adapt = False