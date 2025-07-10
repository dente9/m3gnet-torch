# m3gnet/train/trainer.py (Final Fixed Version for Circular Import)

"""
PyTorch-native trainers for M3GNet models.
"""
from typing import List, Optional, Union, Tuple, Dict
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ase import Atoms
from pymatgen.core import Structure, Molecule

# NOTE: We avoid top-level imports that could cause circular dependencies.
# Specifics like `collate_fn` will be imported within the methods that use them.
from m3gnet.graph import MaterialGraph, RadiusCutoffGraphConverter
from m3gnet.models import M3GNet, Potential
from m3gnet.types import StructureOrMolecule
from .callbacks import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)

class M3GNetDataset(Dataset):
    """
    A simple PyTorch Dataset for M3GNet.
    """
    def __init__(self, structures: List[StructureOrMolecule], targets: List):
        if len(structures) != len(targets):
            raise ValueError("Number of structures and targets must be the same.")
        self.structures = structures
        self.targets = targets
        self.graph_converter = RadiusCutoffGraphConverter(cutoff=5.0)

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        graph = self.graph_converter.convert(self.structures[idx])
        target = self.targets[idx]
        return graph, torch.tensor(target, dtype=torch.float32)

class PotentialDataset(Dataset):
    """
    Dataset for Potential training, handling energies, forces, and stresses.
    """
    def __init__(self, structures: List[StructureOrMolecule], energies: List, forces: List, stresses: Optional[List] = None):
        if not (len(structures) == len(energies) == len(forces)):
            raise ValueError("Mismatch in the number of structures, energies, and forces.")
        if stresses is not None and len(structures) != len(stresses):
            raise ValueError("Mismatch in the number of structures and stresses.")
        
        self.structures = structures
        self.energies = torch.tensor(energies, dtype=torch.float32).view(-1, 1)
        self.forces = [torch.tensor(f, dtype=torch.float32) for f in forces]
        self.stresses = [torch.tensor(s, dtype=torch.float32) for s in stresses] if stresses is not None else None
        self.graph_converter = RadiusCutoffGraphConverter(cutoff=5.0)

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        graph = self.graph_converter.convert(self.structures[idx])
        targets = {"energy": self.energies[idx], "forces": self.forces[idx]}
        if self.stresses is not None and idx < len(self.stresses):
            targets["stress"] = self.stresses[idx]
        return graph, targets


class BaseTrainer:
    """
    Base class for trainers.
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

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
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return {"loss": epoch_loss / len(loader)}

    def _validate_one_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        val_loss = 0.0
        pbar = tqdm(loader, desc="Validation", unit="batch")
        with torch.no_grad():
            for batch in pbar:
                loss = self.calc_loss(batch)
                val_loss += loss.item()
                pbar.set_postfix(val_loss=loss.item())
        return {"val_loss": val_loss / len(loader)}

    def calc_loss(self, batch: Tuple) -> torch.Tensor:
        raise NotImplementedError

# <<<<<<<<<<<<<<<<<<<< THE FIX IS IN THE train() METHODS BELOW <<<<<<<<<<<<<<<<<<<<
class PropertyTrainer(BaseTrainer):
    """
    Trainer for a general property model (e.g., band gap).
    """
    def train(
        self,
        train_structures: List[StructureOrMolecule],
        train_targets: List,
        val_structures: Optional[List[StructureOrMolecule]] = None,
        val_targets: Optional[List] = None,
        batch_size: int = 32,
        epochs: int = 100,
        loss_fn=F.l1_loss,
        callbacks: Optional[List] = None,
    ):
        # Local import to prevent circular dependency at module load time
        from m3gnet.graph.batch import collate_list_of_graphs

        train_dataset = M3GNetDataset(train_structures, train_targets)
        train_dataset.graph_converter = self.model.graph_converter
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_list_of_graphs)

        val_loader = None
        if val_structures and val_targets:
            val_dataset = M3GNetDataset(val_structures, val_targets)
            val_dataset.graph_converter = self.model.graph_converter
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_list_of_graphs)

        self.loss_fn = loss_fn

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
                    cb.on_epoch_end(epoch, logs, self.model)
                if any(getattr(cb, 'stop_training', False) for cb in callbacks):
                    print("Early stopping triggered. Ending training.")
                    break

    def calc_loss(self, batch: Tuple[MaterialGraph, torch.Tensor]) -> torch.Tensor:
        graph, targets = batch
        graph = graph.to(self.device)
        targets = targets.to(self.device)
        
        predictions = self.model(graph)
        return self.loss_fn(predictions.view(-1), targets.view(-1))


class PotentialTrainer(BaseTrainer):
    """
    Trainer for a Potential model, including energy, forces, and stress.
    """
    def __init__(self, potential: Potential, optimizer: torch.optim.Optimizer, device: Union[str, torch.device]):
        super().__init__(potential, optimizer, device)
        self.potential = self.model

    def train(
        self,
        train_structures: List, 
        train_energies: List, 
        train_forces: List,
        train_stresses: Optional[List] = None,
        val_structures: Optional[List] = None, 
        val_energies: Optional[List] = None, 
        val_forces: Optional[List] = None, 
        val_stresses: Optional[List] = None,
        batch_size: int = 8, 
        epochs: int = 100,
        energy_weight: float = 1.0, 
        force_weight: float = 1.0, 
        stress_weight: float = 0.1,
        loss_fn=F.l1_loss, 
        callbacks: Optional[List] = None,
    ):
        # Local import to prevent circular dependency at module load time
        from m3gnet.graph.batch import collate_list_of_graphs
        
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.loss_fn = loss_fn
        
        train_dataset = PotentialDataset(train_structures, train_energies, train_forces, train_stresses)
        train_dataset.graph_converter = self.potential.graph_converter
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_list_of_graphs)

        val_loader = None
        if val_structures and val_energies and val_forces:
            val_dataset = PotentialDataset(val_structures, val_energies, val_forces, val_stresses)
            val_dataset.graph_converter = self.potential.graph_converter
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_list_of_graphs)
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            train_logs = self._train_one_epoch(train_loader)
            
            logs = {**train_logs}
            if val_loader:
                val_logs = self._validate_one_epoch(val_loader)
                logs.update(val_logs)
            
            print(f"Epoch {epoch + 1} Summary: ", " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()]))
            
            if callbacks:
                model_to_save = self.potential.model if isinstance(self.potential, Potential) else self.potential
                for cb in callbacks:
                    cb.on_epoch_end(epoch, logs, model_to_save)
                if any(getattr(cb, 'stop_training', False) for cb in callbacks):
                    print("Early stopping triggered. Ending training.")
                    break
    
    def calc_loss(self, batch: Tuple[MaterialGraph, Dict[str, torch.Tensor]]) -> torch.Tensor:
        graph, targets = batch
        graph = graph.to(self.device)
        
        has_stress = "stress" in targets and targets["stress"] is not None
        
        pred_energy, pred_forces, pred_stress = self.potential(graph, compute_forces=True, compute_stress=has_stress)
        
        n_atoms = graph.n_atoms.to(self.device).view(-1, 1)
        target_energy = targets["energy"].to(self.device)
        e_loss = self.loss_fn(pred_energy / n_atoms, target_energy / n_atoms)
        
        target_forces = targets["forces"].to(self.device)
        f_loss = self.loss_fn(pred_forces, target_forces)

        s_loss = torch.tensor(0.0, device=self.device)
        if has_stress:
            target_stress = targets["stress"].to(self.device)
            s_loss = self.loss_fn(pred_stress, target_stress)
        
        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss + self.stress_weight * s_loss
        return total_loss