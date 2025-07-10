# m3gnet/train/callbacks.py

"""Callback-like classes for PyTorch training loops."""

import torch
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class ModelCheckpoint:
    """
    Save the model after every epoch if the validation metric improves.
    """
    def __init__(self, filepath: str = "best_model.pt", monitor: str = "val_loss", mode: str = "min", verbose: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_score = np.inf if mode == "min" else -np.inf
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict, model: torch.nn.Module):
        """
        Checks if the model should be saved at the end of an epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): A dictionary of metrics from the epoch.
            model (torch.nn.Module): The model to save.
        """
        score = logs.get(self.monitor)
        if score is None:
            return

        if (self.mode == "min" and score < self.best_score) or \
           (self.mode == "max" and score > self.best_score):
            if self.verbose:
                print(f"\nEpoch {epoch+1}: {self.monitor} improved from {self.best_score:.4f} to {score:.4f}. Saving model to {self.filepath}")
            self.best_score = score
            torch.save(model.state_dict(), self.filepath)

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

    def on_epoch_end(self, epoch: int, logs: dict):
        """
        Checks if training should be stopped.
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