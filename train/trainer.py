# m3gnet/train/trainer.py (Final Fixed Version with Normalization)

from typing import List, Optional, Union, Tuple, Dict
import logging
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from m3gnet.graph import MaterialGraph, collate_list_of_graphs, collate_potential_graphs
from m3gnet.models import Potential
from .callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

class M3GNetDataset(Dataset):
    """Dataset for property prediction."""
    def __init__(self, graphs: List[MaterialGraph], targets: np.ndarray):
        if len(graphs) != len(targets):
            raise ValueError("Number of graphs and targets must be the same.")
        self.graphs = graphs
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.graphs)
        
    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]

class PotentialDataset(Dataset):
    """Dataset for potential (energy, force, stress) training."""
    def __init__(self, graphs: List[MaterialGraph], energies: np.ndarray, forces: List[np.ndarray], stresses: Optional[List[np.ndarray]] = None):
        if not (len(graphs) == len(energies) == len(forces)):
            raise ValueError("Mismatch in numbers of graphs, energies, and forces.")
        if stresses is not None and len(graphs) != len(stresses):
            raise ValueError("Mismatch in numbers of graphs and stresses.")
            
        self.graphs = graphs
        self.energies = torch.tensor(energies, dtype=torch.float32)
        self.forces = [torch.tensor(f, dtype=torch.float32) for f in forces]
        self.stresses = [torch.tensor(s, dtype=torch.float32) for s in stresses] if stresses is not None else None
        
    def __len__(self):
        return len(self.graphs)
        
    def __getitem__(self, idx):
        targets = {"energy": self.energies[idx], "forces": self.forces[idx]}
        if self.stresses is not None and idx < len(self.stresses):
            targets["stress"] = self.stresses[idx]
        return self.graphs[idx], targets

class BaseTrainer:
    """Base trainer class with common training and validation loops."""
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = None

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def _train_one_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc="Training", unit="batch")
        for batch in pbar:
            self.optimizer.zero_grad()
            loss = self.calc_loss(batch)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return {"loss": epoch_loss / len(loader)}

    def _validate_one_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        val_loss = 0.0
        pbar = tqdm(loader, desc="Validation", unit="batch")
        with torch.no_grad():
            for batch in pbar:
                loss = self.calc_loss(batch)
                val_loss += loss.item()
                pbar.set_postfix(val_loss=f"{loss.item():.4f}")
        return {"val_loss": val_loss / len(loader)}

    def calc_loss(self, batch: Tuple) -> torch.Tensor:
        raise NotImplementedError

class PropertyTrainer(BaseTrainer):
    """Trainer for predicting a single property."""
    def train(
        self, train_graphs: List[MaterialGraph], train_targets: np.ndarray,
        val_graphs: Optional[List[MaterialGraph]] = None, val_targets: Optional[np.ndarray] = None,
        batch_size: int = 32, epochs: int = 100, loss_fn=F.l1_loss, callbacks: Optional[List] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_workers: int = 0, pin_memory: bool = False
    ):
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        
        train_dataset = M3GNetDataset(train_graphs, train_targets)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=collate_list_of_graphs, num_workers=num_workers, pin_memory=pin_memory
        )
        
        val_loader = None
        if val_graphs is not None and val_targets is not None:
            val_dataset = M3GNetDataset(val_graphs, val_targets)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, 
                collate_fn=collate_list_of_graphs, num_workers=num_workers, pin_memory=pin_memory
            )
        
        checkpoint_callback = None
        if callbacks:
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    checkpoint_callback = cb
                    
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            train_logs = self._train_one_epoch(train_loader)
            logs = {**train_logs}
            
            if val_loader:
                val_logs = self._validate_one_epoch(val_loader)
                logs.update(val_logs)
            
            print(f"Epoch {epoch + 1} Summary: ", " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))
            
            if callbacks:
                for cb in callbacks:
                    cb.on_epoch_end(epoch, logs, model=self.model)
                if any(getattr(cb, 'stop_training', False) for cb in callbacks):
                    print("Early stopping triggered. Ending training.")
                    break
        
        if checkpoint_callback and os.path.exists(checkpoint_callback.best_path):
            best_model_weights_path = os.path.join(checkpoint_callback.best_path, "m3gnet.pt")
            print(f"\nTraining finished. Loading best model weights from {best_model_weights_path}")
            self.model.load_state_dict(torch.load(best_model_weights_path, map_location=self.device))
            
    # --- [ THE ONLY MODIFICATION IS HERE ] ---
    def calc_loss(self, batch: Tuple[MaterialGraph, Tuple[torch.Tensor]]) -> torch.Tensor:
        """
        Calculates the loss for a batch.
        If the model is for an extensive property (is_intensive=False),
        it normalizes the loss by the number of atoms.
        """
        graph, (targets,) = batch
        graph = graph.to(self.device)
        targets = targets.to(self.device)
        
        predictions = self.model(graph)
        
        # Check if the model is for an extensive property
        # The .get() method safely handles cases where 'is_intensive' might not be in hparams
        if not self.model.hparams.get('is_intensive', True):
            # It's an extensive property (like total energy). Normalize by number of atoms.
            n_atoms = graph.n_atoms.to(self.device).view(-1, 1)
            # Prevent division by zero for any empty graphs, though unlikely
            n_atoms = torch.clamp(n_atoms, min=1)
            
            # Calculate per-atom loss
            normalized_preds = predictions / n_atoms
            normalized_targets = targets.view(-1, 1) / n_atoms
            
            return self.loss_fn(normalized_preds, normalized_targets)
        else:
            # It's an intensive property. Calculate loss directly.
            return self.loss_fn(predictions.view(-1), targets.view(-1))

class PotentialTrainer(BaseTrainer):
    """Trainer for potentials (energy, forces, stresses)."""
    def __init__(self, potential: Potential, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
        # The model to be trained is the M3GNet model inside the Potential
        super().__init__(potential.model, optimizer, device)
        self.potential = potential.to(device)
    
    def train(
        self, train_graphs: List[MaterialGraph], train_energies: np.ndarray, train_forces: List[np.ndarray],
        train_stresses: Optional[List[np.ndarray]] = None, val_graphs: Optional[List[MaterialGraph]] = None, 
        val_energies: Optional[np.ndarray] = None, val_forces: Optional[List[np.ndarray]] = None, 
        val_stresses: Optional[List[np.ndarray]] = None, batch_size: int = 8, epochs: int = 100,
        energy_weight: float = 1.0, force_weight: float = 1.0, stress_weight: float = 0.1,
        loss_fn=F.l1_loss, callbacks: Optional[List] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_workers: int = 0, pin_memory: bool = False
    ):
        self.scheduler = scheduler
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.loss_fn = loss_fn
        
        train_dataset = PotentialDataset(train_graphs, train_energies, train_forces, train_stresses)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=collate_potential_graphs, num_workers=num_workers, pin_memory=pin_memory
        )
        
        val_loader = None
        if val_graphs is not None and val_energies is not None and val_forces is not None:
            val_dataset = PotentialDataset(val_graphs, val_energies, val_forces, val_stresses)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, 
                collate_fn=collate_potential_graphs, num_workers=num_workers, pin_memory=pin_memory
            )
        
        checkpoint_callback = None
        if callbacks:
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    checkpoint_callback = cb
                    
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            train_logs = self._train_one_epoch(train_loader)
            logs = {**train_logs}
            
            if val_loader:
                # In validation, we need to pass the model, not self.potential
                val_logs = self._validate_one_epoch(val_loader)
                logs.update(val_logs)
            
            print(f"Epoch {epoch + 1} Summary: ", " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))
            
            if callbacks:
                # Pass the underlying M3GNet model to the callback for saving
                model_to_save = self.potential.model
                for cb in callbacks:
                    cb.on_epoch_end(epoch, logs, model=model_to_save)
                if any(getattr(cb, 'stop_training', False) for cb in callbacks):
                    print("Early stopping triggered. Ending training.")
                    break
        
        if checkpoint_callback and os.path.exists(checkpoint_callback.best_path):
            best_model_weights_path = os.path.join(checkpoint_callback.best_path, "m3gnet.pt")
            print(f"\nTraining finished. Loading best model weights from {best_model_weights_path}")
            self.potential.model.load_state_dict(torch.load(best_model_weights_path, map_location=self.device))
            
    def calc_loss(self, batch: Tuple[MaterialGraph, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Calculates the composite loss for potential training."""
        graph, targets = batch
        graph = graph.to(self.device)
        target_energy = targets["energy"].to(self.device)
        target_forces = targets["forces"].to(self.device)
        has_stress = "stress" in targets and targets["stress"] is not None
        target_stress = targets["stress"].to(self.device) if has_stress else None
        
        # Use the potential object for forward pass to get E, F, S
        pred_energy, pred_forces, pred_stress = self.potential(graph, compute_forces=True, compute_stress=has_stress)
        
        # Normalize energy loss by number of atoms
        n_atoms_per_graph = graph.n_atoms.to(self.device).view(-1, 1)
        n_atoms_per_graph = torch.clamp(n_atoms_per_graph, min=1)
        e_loss = self.loss_fn(pred_energy / n_atoms_per_graph, target_energy.view(-1, 1) / n_atoms_per_graph)
        
        # Force loss
        f_loss = self.loss_fn(pred_forces, target_forces)
        
        # Stress loss
        s_loss = torch.tensor(0.0, device=self.device)
        if has_stress and pred_stress is not None:
            s_loss = self.loss_fn(pred_stress, target_stress)
        
        return self.energy_weight * e_loss + self.force_weight * f_loss + self.stress_weight * s_loss