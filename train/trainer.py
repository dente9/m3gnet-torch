# m3gnet/train/trainer.py (Final Version where Model Handles Normalization)

from typing import List, Optional, Union, Tuple, Dict
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from m3gnet.graph import MaterialGraph, collate_list_of_graphs, collate_potential_graphs
from m3gnet.models import Potential
from .callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

# --- Dataset classes (No Changes Needed) ---
class M3GNetDataset(Dataset):
    """
    A PyTorch Dataset for property prediction.
    It provides the graph, the primary target (e.g., interaction energy),
    and the original total energy target for metric calculation.
    """
    def __init__(self, graphs: List[MaterialGraph], targets: np.ndarray, original_targets: Optional[np.ndarray] = None):
        if len(graphs) != len(targets):
            raise ValueError("Number of graphs and targets must be the same.")
        self.graphs = graphs
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.original_targets = torch.tensor(original_targets, dtype=torch.float32) if original_targets is not None else self.targets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> Tuple[MaterialGraph, torch.Tensor, torch.Tensor]:
        return self.graphs[idx], self.targets[idx], self.original_targets[idx]

class PotentialDataset(Dataset):
    """
    A PyTorch Dataset for potential training (energy, forces, stresses).
    """
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

    def __getitem__(self, idx) -> Tuple[MaterialGraph, Dict[str, torch.Tensor]]:
        targets = {"energy": self.energies[idx], "forces": self.forces[idx]}
        if self.stresses is not None and idx < len(self.stresses):
            targets["stress"] = self.stresses[idx]
        return self.graphs[idx], targets


class BaseTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = None
    def train(self, *args, **kwargs): raise NotImplementedError
    def _train_one_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        epoch_metrics = {}
        pbar = tqdm(loader, desc="Training", unit="batch")
        for batch in pbar:
            self.optimizer.zero_grad()
            batch_metrics = self.calc_loss_and_metrics(batch)
            loss = batch_metrics['loss']
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None: self.scheduler.step()
            for key, value in batch_metrics.items():
                if key not in epoch_metrics: epoch_metrics[key] = 0.0
                epoch_metrics[key] += value.item()
            pbar.set_postfix({k: f"{v.item():.4f}" for k, v in batch_metrics.items()})
        return {k: v / len(loader) for k, v in epoch_metrics.items()}
    def _validate_one_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        epoch_metrics = {}
        pbar = tqdm(loader, desc="Validation", unit="batch")
        with torch.no_grad():
            for batch in pbar:
                batch_metrics = self.calc_loss_and_metrics(batch)
                for key, value in batch_metrics.items():
                    val_key = f"val_{key}"
                    if val_key not in epoch_metrics: epoch_metrics[val_key] = 0.0
                    epoch_metrics[val_key] += value.item()
                pbar.set_postfix({f"val_{k}": f"{v.item():.4f}" for k, v in batch_metrics.items()})
        return {k: v / len(loader) for k, v in epoch_metrics.items()}
    def calc_loss_and_metrics(self, batch: Tuple) -> Dict[str, torch.Tensor]: raise NotImplementedError

class PropertyTrainer(BaseTrainer):
    # This trainer is now simplified. It expects normalized targets (if applicable)
    # and relies on the model to handle its internal un-normalization for metrics.
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
        super().__init__(model, optimizer, device)

    def train(
        self, train_graphs: List[MaterialGraph], train_targets: np.ndarray,
        val_graphs: Optional[List[MaterialGraph]] = None, val_targets: Optional[np.ndarray] = None,
        train_original_targets: Optional[np.ndarray] = None, val_original_targets: Optional[np.ndarray] = None,
        batch_size: int = 32, epochs: int = 100, loss_fn=F.l1_loss, callbacks: Optional[List] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_workers: int = 0, pin_memory: bool = False
    ):
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        train_dataset = M3GNetDataset(train_graphs, train_targets, train_original_targets)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=collate_list_of_graphs, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = None
        if val_graphs is not None and val_targets is not None:
            val_dataset = M3GNetDataset(val_graphs, val_targets, val_original_targets)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, 
                collate_fn=collate_list_of_graphs, num_workers=num_workers, pin_memory=pin_memory
            )
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
        if callbacks:
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint) and os.path.exists(cb.best_path):
                    best_model_weights_path = os.path.join(cb.best_path, "m3gnet.pt")
                    print(f"\nTraining finished. Loading best model weights from {best_model_weights_path}")
                    self.model.load_state_dict(torch.load(best_model_weights_path, map_location=self.device))
                    break

    # <<<<<<<<<<<<<<<<< THE FINAL, CORRECT IMPLEMENTATION <<<<<<<<<<<<<<<<<
    def calc_loss_and_metrics(self, batch: Tuple[MaterialGraph, Tuple[torch.Tensor, ...]]) -> Dict[str, torch.Tensor]:
        graph, targets_tuple = batch
        # This is the NORMALIZED interaction energy
        normalized_interaction_targets, original_total_targets = targets_tuple[0], targets_tuple[1]
        
        graph = graph.to(self.device)
        normalized_interaction_targets = normalized_interaction_targets.to(self.device)
        original_total_targets = original_total_targets.to(self.device)
        
        # The model's forward pass returns the PREDICTED TOTAL ENERGY
        pred_total_energies = self.model(graph)
        metrics = {}
        
        if not self.model.hparams.get('is_intensive', True):
            # To get the predicted NORMALIZED interaction energy, we must reverse the model's forward pass logic
            element_ref_energies = self.model.element_ref_calc(graph)
            # 1. Get predicted interaction energy (un-normalized)
            pred_interaction_energies = pred_total_energies - element_ref_energies
            # 2. Normalize it using the model's stored mean and std
            pred_normalized_interaction = (pred_interaction_energies - self.model.mean) / self.model.std

            n_atoms = graph.n_atoms.to(self.device).view(-1, 1).clamp(min=1)

            # `loss` is calculated on the NORMALIZED, PER-ATOM interaction energy
            loss_preds = pred_normalized_interaction / n_atoms
            loss_targets = normalized_interaction_targets.view(-1, 1) / n_atoms
            metrics['loss'] = self.loss_fn(loss_preds, loss_targets)
            
            # `mae` is calculated on the TOTAL energy, PER-ATOM for interpretability
            mae_per_structure = torch.abs(pred_total_energies - original_total_targets.view(-1, 1))
            metrics['mae'] = (mae_per_structure / n_atoms).mean()
        else:
            metrics['loss'] = self.loss_fn(pred_total_energies.squeeze(), original_total_targets.squeeze())
            metrics['mae'] = metrics['loss'].clone().detach()

        return metrics

class PotentialTrainer(BaseTrainer):
    """
    Trainer for M3GNet potentials, training on energy, forces, and optionally stresses.
    """
    def __init__(self, potential: Potential, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
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
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            train_logs = self._train_one_epoch(train_loader)
            logs = {**train_logs}
            
            if val_loader:
                val_logs = self._validate_one_epoch(val_loader)
                logs.update(val_logs)
            
            print(f"Epoch {epoch + 1} Summary: ", " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))
            
            if callbacks:
                model_to_save = self.potential.model
                for cb in callbacks:
                    cb.on_epoch_end(epoch, logs, model=model_to_save)
                if any(getattr(cb, 'stop_training', False) for cb in callbacks):
                    print("Early stopping triggered. Ending training.")
                    break
        
        if callbacks:
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint) and os.path.exists(cb.best_path):
                    best_model_weights_path = os.path.join(cb.best_path, "m3gnet.pt")
                    print(f"\nTraining finished. Loading best model weights from {best_model_weights_path}")
                    self.potential.model.load_state_dict(torch.load(best_model_weights_path, map_location=self.device))
                    break
            
    def calc_loss_and_metrics(self, batch: Tuple[MaterialGraph, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        graph, targets = batch
        graph = graph.to(self.device)
        target_energy = targets["energy"].to(self.device)
        target_forces = targets["forces"].to(self.device)
        has_stress = "stress" in targets and targets["stress"] is not None
        target_stress = targets["stress"].to(self.device) if has_stress else None
        
        pred_energy, pred_forces, pred_stress = self.potential(graph, compute_forces=True, compute_stress=has_stress)
        
        n_atoms_per_graph = graph.n_atoms.to(self.device).view(-1, 1).clamp(min=1)
        # Potential trainer loss is also on interaction energy
        e_loss = self.loss_fn(pred_energy.view(-1, 1) / n_atoms_per_graph, target_energy.view(-1, 1) / n_atoms_per_graph)
        
        f_loss = self.loss_fn(pred_forces, target_forces)
        
        s_loss = torch.tensor(0.0, device=self.device)
        if has_stress and pred_stress is not None:
            s_loss = self.loss_fn(pred_stress, target_stress)
        
        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss + self.stress_weight * s_loss
        
        return {'loss': total_loss, 'e_loss': e_loss, 'f_loss': f_loss, 's_loss': s_loss}