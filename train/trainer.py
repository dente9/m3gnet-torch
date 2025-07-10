# m3gnet/train/trainer.py (Final Attribute-Fixed Version)
from typing import List, Optional, Union, Tuple, Dict
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ase import Atoms
from pymatgen.core import Structure, Molecule
import numpy as np

from m3gnet.graph.batch import collate_list_of_graphs, collate_potential_graphs
from m3gnet.graph import MaterialGraph, RadiusCutoffGraphConverter, StructureOrMolecule
from m3gnet.models import M3GNet, Potential
from .callbacks import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)

# <<<<<<<<<<<<<<<<<<<< THE FIX IS IN THE __init__ METHODS BELOW <<<<<<<<<<<<<<<<<<<<
class M3GNetDataset(Dataset):
    def __init__(self, structures: List[StructureOrMolecule], targets: np.ndarray):
        if len(structures) != len(targets): raise ValueError("Number of structures and targets must be the same.")
        self.structures = structures
        self.targets = torch.tensor(targets, dtype=torch.float32)
        # The converter instance needs to be stored on the object itself.
        self.converter = RadiusCutoffGraphConverter(cutoff=5.0)
    def __len__(self): return len(self.structures)
    def __getitem__(self, idx):
        # Now self.converter exists.
        graph = self.converter.convert(self.structures[idx])
        return graph, self.targets[idx]

class PotentialDataset(Dataset):
    def __init__(self, structures: List[StructureOrMolecule], energies: np.ndarray, forces: List[np.ndarray], stresses: Optional[List[np.ndarray]] = None):
        if not (len(structures) == len(energies) == len(forces)): raise ValueError("Mismatch in the number of structures, energies, and forces.")
        if stresses is not None and len(structures) != len(stresses): raise ValueError("Mismatch in the number of structures and stresses.")
        self.structures = structures
        self.energies = torch.tensor(energies, dtype=torch.float32)
        self.forces = [torch.tensor(f, dtype=torch.float32) for f in forces]
        self.stresses = [torch.tensor(s, dtype=torch.float32) for s in stresses] if stresses is not None else None
        # The converter instance needs to be stored on the object itself.
        self.converter = RadiusCutoffGraphConverter(cutoff=5.0)
    def __len__(self): return len(self.structures)
    def __getitem__(self, idx):
        # Now self.converter exists.
        graph = self.converter.convert(self.structures[idx])
        targets = {"energy": self.energies[idx], "forces": self.forces[idx]}
        if self.stresses is not None and idx < len(self.stresses): targets["stress"] = self.stresses[idx]
        return graph, targets

# The rest of the file is correct and does not need changes.
class BaseTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
    def train(self, *args, **kwargs): raise NotImplementedError
    def _train_one_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc="Training", unit="batch")
        for batch in pbar:
            self.optimizer.zero_grad()
            loss = self.calc_loss(batch)
            loss.backward()
            self.optimizer.step()
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
    def calc_loss(self, batch: Tuple) -> torch.Tensor: raise NotImplementedError

class PropertyTrainer(BaseTrainer):
    def train(
        self, train_structures: List[StructureOrMolecule], train_targets: np.ndarray,
        val_structures: Optional[List[StructureOrMolecule]] = None, val_targets: Optional[np.ndarray] = None,
        batch_size: int = 32, epochs: int = 100, loss_fn=F.l1_loss, callbacks: Optional[List] = None,
    ):
        train_dataset = M3GNetDataset(train_structures, train_targets)
        # Allow trainer's model's converter to override the default dataset converter
        if hasattr(self.model, 'graph_converter'):
            train_dataset.converter = self.model.graph_converter
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_list_of_graphs)
        
        val_loader = None
        if val_structures is not None and val_targets is not None:
            val_dataset = M3GNetDataset(val_structures, val_targets)
            if hasattr(self.model, 'graph_converter'):
                val_dataset.converter = self.model.graph_converter
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_list_of_graphs)
        
        self.loss_fn = loss_fn
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
            print(f"\nTraining finished. Loading best model from {checkpoint_callback.best_path}")
            self.model.load_state_dict(torch.load(checkpoint_callback.best_path, map_location=self.device))
            
    def calc_loss(self, batch: Tuple[MaterialGraph, Tuple[torch.Tensor]]) -> torch.Tensor:
        graph, (targets,) = batch
        graph = graph.to(self.device)
        targets = targets.to(self.device)
        predictions = self.model(graph)
        return self.loss_fn(predictions.view(-1), targets.view(-1))

class PotentialTrainer(BaseTrainer):
    def __init__(self, potential: Potential, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
        super().__init__(potential, optimizer, device)
        self.potential = self.model
    
    def train(
        self, train_structures: List, train_energies: np.ndarray, train_forces: List[np.ndarray],
        train_stresses: Optional[List[np.ndarray]] = None, val_structures: Optional[List] = None, 
        val_energies: Optional[np.ndarray] = None, val_forces: Optional[List[np.ndarray]] = None, 
        val_stresses: Optional[List[np.ndarray]] = None, batch_size: int = 8, epochs: int = 100,
        energy_weight: float = 1.0, force_weight: float = 1.0, stress_weight: float = 0.1,
        loss_fn=F.l1_loss, callbacks: Optional[List] = None,
    ):
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.loss_fn = loss_fn
        
        train_dataset = PotentialDataset(train_structures, train_energies, train_forces, train_stresses)
        train_dataset.converter = self.potential.graph_converter
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_potential_graphs)
        
        val_loader = None
        if val_structures is not None and val_energies is not None and val_forces is not None:
            val_dataset = PotentialDataset(val_structures, val_energies, val_forces, val_stresses)
            val_dataset.converter = self.potential.graph_converter
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_potential_graphs)
        
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
                model_to_save = self.potential.model
                for cb in callbacks:
                    cb.on_epoch_end(epoch, logs, model=model_to_save)
                if any(getattr(cb, 'stop_training', False) for cb in callbacks):
                    print("Early stopping triggered. Ending training.")
                    break
        
        if checkpoint_callback and os.path.exists(checkpoint_callback.best_path):
            print(f"\nTraining finished. Loading best model from {checkpoint_callback.best_path}")
            self.potential.model.load_state_dict(torch.load(checkpoint_callback.best_path, map_location=self.device))
            
    def calc_loss(self, batch: Tuple[MaterialGraph, Dict[str, torch.Tensor]]) -> torch.Tensor:
        graph, targets = batch
        graph = graph.to(self.device)
        target_energy = targets["energy"].to(self.device)
        target_forces = targets["forces"].to(self.device)
        has_stress = "stress" in targets and targets["stress"] is not None
        target_stress = targets["stress"].to(self.device) if has_stress else None
        
        pred_energy, pred_forces, pred_stress = self.potential(graph, compute_forces=True, compute_stress=has_stress)
        
        n_atoms_per_graph = graph.n_atoms.to(self.device).view(-1, 1)
        e_loss = self.loss_fn(pred_energy / n_atoms_per_graph, target_energy / n_atoms_per_graph)
        
        f_loss = self.loss_fn(pred_forces, target_forces)
        
        s_loss = torch.tensor(0.0, device=self.device)
        if has_stress and pred_stress is not None:
            s_loss = self.loss_fn(pred_stress, target_stress)
        
        return self.energy_weight * e_loss + self.force_weight * f_loss + self.stress_weight * s_loss