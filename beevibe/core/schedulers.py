import torch

class AnnealOnPlateau:
    """
    Implements a learning rate scheduler that reduces the learning rate when a metric has stopped improving.

    Attributes:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        initial_extra_patience (int): Extra patience for the initial phase.
        mode (str): One of {"min", "max"}. In "min" mode, learning rate will be reduced when the quantity monitored has stopped decreasing; in "max" mode it will be reduced when the quantity monitored has stopped increasing.
        min_lr (float): A lower bound on the learning rate.
        verbose (bool): If True, prints a message to stdout for each update.
        num_bad_epochs (int): Number of epochs with no improvement.
        best (float): Best metric value seen so far.
        cooldown_counter (int): Cooldown counter to delay further learning rate reductions.
        cooldown (int): Cooldown epochs after learning rate reduction.
        mode_worse (float): Value worse than the best for the chosen mode.
        is_better (callable): Function to compare metrics based on the mode.
    """

    def __init__(
        self,
        optimizer: "torch.optim.Optimizer",
        factor: float = 0.5,
        patience: int = 3,
        initial_extra_patience: int = 0,
        mode: str = "min",
        min_lr: float = 0.0001,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the AnnealOnPlateau scheduler.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            factor (float, optional): Factor by which the learning rate will be reduced. Default is 0.5.
            patience (int, optional): Number of epochs with no improvement before reducing the learning rate. Default is 3.
            initial_extra_patience (int, optional): Extra patience for the initial phase. Default is 0.
            mode (str, optional): One of {"min", "max"}. Default is "min".
            min_lr (float, optional): Minimum learning rate. Default is 0.0001.
            verbose (bool, optional): If True, prints a message to stdout for each update. Default is False.
        """
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

    def step(self, metrics: float) -> None:
        """
        Update learning rate based on the current metrics.

        Args:
            metrics (float): The current value of the monitored metric.
        """
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # Ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown

    def _reduce_lr(self) -> None:
        """
        Reduce the learning rate by the specified factor.
        """
        for param_group in self.optimizer.param_groups:
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > 1e-8:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate to {new_lr}")

    @property
    def in_cooldown(self) -> bool:
        """
        Check if the scheduler is in the cooldown phase.

        Returns:
            bool: True if in cooldown, False otherwise.
        """
        return self.cooldown_counter > 0
