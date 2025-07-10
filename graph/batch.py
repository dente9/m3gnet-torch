# m3gnet/graph/batch.py (Final-Final Version)
"""
Tools for batching MaterialGraph objects for training.
"""
from typing import List, Tuple, Optional

import torch
import numpy as np
from torch.utils.data import Dataset

from .struct_to_graph import MaterialGraph, RadiusCutoffGraphConverter, StructureOrMolecule

def collate_list_of_graphs(batch: List[Tuple[MaterialGraph, ...]]) -> Tuple[MaterialGraph, Tuple[torch.Tensor, ...]]:
    if not batch:
        return None, None

    graphs = [item[0] for item in batch]
    targets_list = [item[1:] for item in batch]

    # Batch graph attributes
    atom_features = torch.cat([g.atom_features for g in graphs], dim=0)
    bond_features = torch.cat([g.bond_features for g in graphs], dim=0)
    atom_positions = torch.cat([g.atom_positions for g in graphs], dim=0)
    pbc_offsets = torch.cat([g.pbc_offsets for g in graphs], dim=0)
    
    atom_offsets = torch.cumsum(torch.tensor([0] + [g.n_atoms.item() for g in graphs[:-1]]), dim=0)
    bond_atom_indices = torch.cat([g.bond_atom_indices + offset for g, offset in zip(graphs, atom_offsets)], dim=0)
    
    batched_graph = MaterialGraph(
        atom_features=atom_features, bond_features=bond_features,
        atom_positions=atom_positions, bond_atom_indices=bond_atom_indices,
        pbc_offsets=pbc_offsets, n_atoms=torch.tensor([g.n_atoms.item() for g in graphs]),
        n_bonds=torch.tensor([g.n_bonds.item() for g in graphs]),
        lattices=torch.cat([g.lattices for g in graphs if g.lattices is not None]) if any(g.lattices is not None for g in graphs) else None,
        state_features=torch.cat([g.state_features for g in graphs if g.state_features is not None]) if any(g.state_features is not None for g in graphs) else None,
        has_three_body=any(g.has_three_body for g in graphs)
    )

    if batched_graph.has_three_body:
        bond_offsets = torch.cumsum(torch.tensor([0] + [g.n_bonds.item() for g in graphs[:-1]]), dim=0)
        triple_bond_indices = [g.triple_bond_indices + offset for g, offset in zip(graphs, bond_offsets) if g.triple_bond_indices is not None]
        batched_graph.triple_bond_indices = torch.cat(triple_bond_indices) if triple_bond_indices else None
        triple_features = [g.triple_features for g in graphs if g.triple_features is not None]
        batched_graph.triple_features = torch.cat(triple_features) if triple_features else None
        triple_bond_lengths = [g.triple_bond_lengths for g in graphs if g.triple_bond_lengths is not None]
        batched_graph.triple_bond_lengths = torch.cat(triple_bond_lengths) if triple_bond_lengths else None
        n_triples = [g.n_triples.item() for g in graphs if g.n_triples is not None]
        batched_graph.n_triples = torch.tensor(n_triples) if n_triples else torch.tensor([0])
    
    # Batch targets
    batched_targets = []
    if targets_list and targets_list[0]:
        num_targets_per_graph = len(targets_list[0])
        for i in range(num_targets_per_graph):
            current_targets = [t[i] for t in targets_list]
            if all(t is None for t in current_targets):
                batched_targets.append(None)
                continue
            
            # Check if it's a per-atom property (like forces) vs per-graph (like energy)
            if current_targets[0].ndim > 0 and current_targets[0].shape[0] == graphs[0].n_atoms.item():
                batched_targets.append(torch.cat(current_targets, dim=0))
            else:
                # <<<<<<<<<<<<<<<<<<<< THE FIX IS HERE <<<<<<<<<<<<<<<<<<<<
                # Ensure per-graph properties are always [batch_size, 1] for consistency
                stacked_targets = torch.stack(current_targets, dim=0)
                batched_targets.append(stacked_targets.view(-1, 1))
    
    return batched_graph, tuple(batched_targets)


class MaterialGraphDataset(Dataset):
    def __init__(
        self,
        structures: List[StructureOrMolecule],
        converter: RadiusCutoffGraphConverter,
        energies: Optional[List[float]] = None,
        forces: Optional[List[np.ndarray]] = None,
        stresses: Optional[List[np.ndarray]] = None,
    ):
        self.structures = structures
        self.converter = converter
        self.energies = energies
        self.forces = forces
        self.stresses = stresses

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, index: int) -> Tuple[MaterialGraph, ...]:
        struct = self.structures[index]
        graph = self.converter.convert(struct)
        
        targets = []
        if self.energies is not None:
            targets.append(torch.tensor(self.energies[index], dtype=torch.float32))
        if self.forces is not None:
            targets.append(torch.tensor(self.forces[index], dtype=torch.float32))
        if self.stresses is not None:
            targets.append(torch.tensor(self.stresses[index], dtype=torch.float32))
            
        return (graph, *targets)