# m3gnet/train/callbacks.py (Final Version)

"""Callback-like classes for PyTorch training loops."""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class ModelCheckpoint:
    """
    Save the model after every epoch if the validation metric improves.
    This callback calls the model's own .save() method.
    """
    def __init__(
        self, 
        save_dir: str = "checkpoints", 
        monitor: str = "val_loss", 
        mode: str = "min", 
        best_dirname: str = "best_model",
        last_dirname: str = "last_model",
        verbose: bool = True
    ):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        # These paths now point to the directories where models will be saved
        self.best_path = os.path.join(self.save_dir, best_dirname)
        self.last_path = os.path.join(self.save_dir, last_dirname)
        
        self.best_score = np.inf if mode == "min" else -np.inf
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict, model):
        """
        Saves the model by calling its .save() method.
        The model object passed in should have a .save(directory) method.
        """
        # Save the latest model
        model.save(self.last_path)

        score = logs.get(self.monitor)
        if score is None:
            return

        if (self.mode == "min" and score < self.best_score) or \
           (self.mode == "max" and score > self.best_score):
            if self.verbose:
                print(f"\nEpoch {epoch+1}: {self.monitor} improved from {self.best_score:.4f} to {score:.4f}. Saving best model to {self.best_path}")
            self.best_score = score
            # Save the best model
            model.save(self.best_path)

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
        score = logs.get(self.monitor)
        if score is None: return

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