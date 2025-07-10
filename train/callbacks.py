# m3gnet/train/callbacks.py (Final Version with Multiple Saves)

"""Callback-like classes for PyTorch training loops."""

import torch
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class ModelCheckpoint:
    """
    Save the model after every epoch.
    
    This callback saves:
    - `best_model.pt`: The model with the best validation metric so far.
    - `last_model.pt`: The model from the very last epoch.
    """
    def __init__(
        self, 
        save_dir: str = "checkpoints", 
        monitor: str = "val_loss", 
        mode: str = "min", 
        best_filename: str = "best_model.pt",
        last_filename: str = "last_model.pt",
        verbose: bool = True
    ):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_filename = best_filename
        self.last_filename = last_filename
        
        self.best_path = os.path.join(self.save_dir, self.best_filename)
        self.last_path = os.path.join(self.save_dir, self.last_filename)
        
        self.best_score = np.inf if mode == "min" else -np.inf
        
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict, model: torch.nn.Module):
        """
        Checks if the model should be saved at the end of an epoch.
        """
        # Save the latest model
        torch.save(model.state_dict(), self.last_path)

        score = logs.get(self.monitor)
        if score is None:
            return

        if (self.mode == "min" and score < self.best_score) or \
           (self.mode == "max" and score > self.best_score):
            if self.verbose:
                print(f"\nEpoch {epoch+1}: {self.monitor} improved from {self.best_score:.4f} to {score:.4f}. Saving best model to {self.best_path}")
            self.best_score = score
            torch.save(model.state_dict(), self.best_path)

class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.
    """
    def __init__(self, monitor: str = "val_loss", mode: str = "min", patience: int = 10, verbose: bool = True):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.wait_counter = 0
        self.best_score = np.inf if mode == "min" else -np.inf
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: dict, **kwargs):
        """
        Checks if training should be stopped. Ignores extra kwargs.
        """
        score = logs.get(self.monitor)
        if score is None:
            return

        if (self.mode == "min" and score < self.best_score) or \
           (self.mode == "max" and score > self.best_score):
            self.best_score = score
            self.wait_counter = 0
        else:
            self.wait_counter += 1
            if self.wait_counter >= self.patience:
                if self.verbose:
                    print(f"\nEpoch {epoch+1}: Early stopping triggered after {self.patience} epochs of no improvement.")
                self.stop_training = True