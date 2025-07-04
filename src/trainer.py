import time
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from hooks import BaseHook
from typing import List
from torch import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_


class Trainer:
    """Trainer class for managing training processes."""

    # TODO: use abc and dataclass
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        loss_fn: nn.Module,
        device: str = "cuda",
        verbose: bool = False,
        epochs: int = 10,
        checkpoint_path: str = "models",
        checkpoint_period: int = 1,
        enable_amp: bool = True,
        clip_grad_norm: float = 0.0,  # TODO
    ):
        """Initialize the Trainer."""
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.verbose = verbose
        self.epochs = epochs
        self._hooks: List[BaseHook] = []
        self._data_iter = iter(self.train_loader)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_period = checkpoint_period
        self.enable_amp = enable_amp
        self._clip_grad_norm = clip_grad_norm
        self.epoch_len: int = len(self.train_loader)
        self.star_iter: int = 0
        self.cur_iter: int = 0
        self.max_iters: int = self.epochs * self.epoch_len
        self._grad_scaler = GradScaler(enabled=self._enable_amp)
        self._default_setup()

    @property
    def lr(self) -> float:
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    @property
    def cur_epoch(self) -> int:
        """Get the current epoch."""
        return self.star_iter // self.epoch_len

    def _default_setup(self) -> None:
        pass

    def register_hooks(self, hooks: List[BaseHook]) -> None:
        """Registers multiple hooks

        Args:
            hooks (List[BaseHook]): List of hook instances to register.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook: BaseHook) -> None:
        """Registers a hook into the hook list, maintaining the order based on priority.

        Hooks with higher priority values are placed later in the list. If multiple hooks have the same priority,
        the new hook is inserted after existing hooks with the same priority.

        Args:
            hook (BaseHook): The hook instance to be registered.
        """
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if hook.priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def _call_hooks(self, stage: str, *args, **kwargs) -> None:
        """Call hooks with the given name."""
        for hook in self._hooks:
            if hasattr(hook, stage):
                getattr(hook, stage)(*args, **kwargs)

    def train(self) -> None:
        """Start the training process."""
        logger.info("Starting training...")
        self._call_hooks("before_train")
        for self.cur_iter in range(self.start_iter, self.max_iters):
            if self.cur_iter % self.epoch_len == 0:
                self._call_hooks("before_epoch")
            self._call_hooks("before_iter")
            self.train_one_iter()
            self._call_hooks("after_iter")
            if (self.cur_iter + 1) % self.epoch_len == 0:
                self._call_hooks("after_epoch")
        self._call_hooks("after_train")
        logger.info("Training completed.")

    def train_one_iter(self) -> None:
        """Train the model for one iteration."""
        iter_start_time = time.perf_counter()

        ######################
        # 1. Load batch data #
        ######################

        start = time.perf_counter()
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            batch = next(self._data_iter)
        data_time = time.perf_counter() - start

        #####################
        # 2. Calculate loss #
        #####################
        # If self._enable_amp=False, autocast and GradScaler’s calls become no-ops.
        # This allows switching between default precision and mixed precision
        # without if-else statements.
        with autocast(enabled=self._enable_amp):
            losses = self.model(batch)

        ##########################
        # 3. Calculate gradients #
        ##########################
        self.optimizer.zero_grad()
        self._grad_scaler.scale(losses).backward()
        if self._clip_grad_norm > 0:
            self._grad_scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)

        ##############################
        # 4. Update model parameters #
        ##############################
        self._grad_scaler.step(self.optimizer)
        self._grad_scaler.update()

        self._log_iter_metrics(losses, data_time, time.perf_counter() - iter_start_time)

    def save_checkpoint(self, epoch: int) -> None:
        pass

    def _log_iter_metrics(self, losses: float, data_time: float, iter_time: float) -> None:
        """Log metrics for the current iteration."""
        if self.verbose:
            logger.info(
                f"Epoch [{self.cur_epoch + 1}/{self.epochs}], "
                f"Iter [{self.cur_iter + 1}/{self.max_iters}], "
                f"Loss: {losses:.4f}, "
                f"Data Time: {data_time:.4f}s, "
                f"Iter Time: {iter_time:.4f}s, "
                f"LR: {self.lr:.6f}"
            )