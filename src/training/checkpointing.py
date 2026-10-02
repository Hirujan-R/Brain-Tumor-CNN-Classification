import os
from pathlib import Path

import torch
import torch.nn as nn


class ModelCheckpoint:
    """
    Saves the best PyTorch model across training epochs based on a monitored metric.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Args:
            filepath (str): The path to save the .pth file.
            monitor (str): The metric being monitored (e.g. 'val_loss').
            mode (str): 'min' or 'max'. If 'min', the model is saved when the
                metric decreases. If 'max', it is saved when the metric increases.
            verbose (bool): Whether to print save events.
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.verbose = verbose

        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        self.mode = mode

        # Initialize best metric
        self.best_metric = float("inf") if mode == "min" else -float("inf")
        self.best_epoch = 0

    def step(
        self,
        current_metric: float,
        model: nn.Module,
        epoch: int,
        optimizer: torch.optim.Optimizer = None,
        scheduler=None,
        save_optimizer: bool = False,
    ) -> bool:
        """
        Checks if the current metric improves upon the best metric. If so, saves
        the model to `self.filepath`.

        Returns:
            bool: True if the model was saved, False otherwise.
        """
        is_best = False
        if self.mode == "min" and current_metric < self.best_metric:
            is_best = True
        elif self.mode == "max" and current_metric > self.best_metric:
            is_best = True

        if not is_best:
            return False

        if self.verbose:
            print(
                f"[Checkpoint] Epoch {epoch}: {self.monitor} improved from "
                f"{self.best_metric:.4f} to {current_metric:.4f}. Saving to {self.filepath}"
            )

        self.best_metric = current_metric
        self.best_epoch = epoch

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_metric": self.best_metric,
            "monitor": self.monitor,
        }
        if save_optimizer and optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if save_optimizer and scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, self.filepath)
        return True
