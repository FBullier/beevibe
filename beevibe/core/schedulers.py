class AnnealOnPlateau:
    def __init__(
        self,
        optimizer,
        factor=0.5,
        patience=3,
        initial_extra_patience=0,
        mode="min",
        min_lr=0.0001,
        verbose=False,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.initial_extra_patience = initial_extra_patience
        self.mode = mode
        self.min_lr = min_lr
        self.verbose = verbose
        self.num_bad_epochs = 0
        self.best = None
        self.cooldown_counter = 0  # Cooldown counter
        self.cooldown = 0  # Cooldown epochs after lr reduction
        self.mode_worse = float("inf") if mode == "min" else -float("inf")
        self.is_better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        self.best = self.mode_worse

    def step(self, metrics):
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > 1e-8:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate to {new_lr}")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
