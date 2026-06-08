import os
import torch
import torch.nn as nn
from typing import Dict, Any

class ModelCheckpoint:
    """
    Saves the best PyTorch model across training epochs based on a monitored metric.
    """
    def __init__(
        self, 
        filepath: str, 
        monitor: str = 'val_loss', 
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            filepath (str): The path to save the .pth file.
            monitor (str): The metric to monitor (e.g., 'val_loss', 'val_accuracy').
            mode (str): 'min' or 'max'. If 'min', model is saved when metric decreases. 
                        If 'max', model is saved when metric increases.
            verbose (bool): Whether to print save events.
        """
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        self.mode = mode
        
        # Initialize best metric
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)

    def step(
        self, 
        current_metric: float, 
        model: nn.Module, 
        epoch: int, 
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None
    ) -> bool:
        """
        Checks if the current metric improves upon the best metric. If so, saves the model.
        
        Args:
            current_metric (float): The current value of the monitored metric.
            model (nn.Module): The model to save.
            epoch (int): The current epoch number.
            optimizer: Optional optimizer state to save for resuming training.
            scheduler: Optional scheduler state to save.
            
        Returns:
            bool: True if model was saved, False otherwise.
        """
        is_best = False
        if self.mode == 'min' and current_metric < self.best_metric:
            is_best = True
        elif self.mode == 'max' and current_metric > self.best_metric:
            is_best = True
            
        if is_best:
            if self.verbose:
                print(f"[Checkpoint] Epoch {epoch}: {self.monitor} improved from {self.best_metric:.4f} to {current_metric:.4f}. Saving model to {self.filepath}")
            
            self.best_metric = current_metric
            
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_metric': self.best_metric,
                'monitor': self.monitor
            }
            if optimizer is not None:
                state['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler is not None:
                state['scheduler_state_dict'] = scheduler.state_dict()
                
            torch.save(state, self.filepath)
            return True
            
        return False
