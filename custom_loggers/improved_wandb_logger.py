from avalanche.logging import WandBLogger


class ImprovedWandBLogger(WandBLogger):
    def __init__(self, model=None, model_log_freq: int = 30, *args, **kwargs):
        self.model = model
        self.model_log_freq = model_log_freq
        super().__init__(*args, **kwargs)

    def before_run(self):
        super().before_run()
        if self.model is not None:
            self.wandb.watch(self.model, log_freq=self.model_log_freq)