class BeeEarlyStopping:
    """
    Implements early stopping mechanism to halt training when validation performance stops improving.

    Attributes:
        patience (int): Number of consecutive epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored value to qualify as an improvement.
        best_loss (float or None): The best validation loss encountered so far.
        counter (int): Number of consecutive epochs with no significant improvement.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """
        Initialize EarlyStopping with specified patience and minimum delta.

        Args:
            patience (int, optional): Number of epochs with no improvement to wait before stopping. Default is 3.
            min_delta (float, optional): Minimum change to consider as an improvement. Default is 0.0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        """
        Check if training should be stopped based on validation loss.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
