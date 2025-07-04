class BaseHook:
    """Base class for hooks in the Trainer.
    Each hook has a priority, which is an integer from 1 to 10.
    The smaller the number, the higher the priority.
    Hooks with the same priority are executed in the order they are registered.
    """

    trainer: "Trainer" = None
    priority: int = 5

    def before_train(self) -> None:
        """Called before training starts."""
        pass

    def after_train(self) -> None:
        """Called after training ends."""
        pass

    def before_epoch(self) -> None:
        """Called before each epoch starts."""
        pass

    def after_epoch(self) -> None:
        """Called after each epoch ends."""
        pass

    def before_iter(self) -> None:
        """Called before each iteration starts."""
        pass

    def after_iter(self) -> None:
        """Called after each iteration ends."""
        pass

    def every_n_epoch(self, n: int) -> bool:
        """Called every n epochs.

        Args:
            n (int): The number of epochs after which this hook is called.

        Returns:
            bool: True if the hook should be executed, False otherwise.
        """
        return (self.trainer.cur_epoch + 1) % n == 0 if n > 0 else False

    def is_last_epoch(self) -> bool:
        """Check if the current epoch is the last one.

        Returns:
            bool: True if the current epoch is the last one, False otherwise.
        """
        return self.trainer.cur_epoch == self.trainer.epochs - 1
