# m3gnet/graph/batch.py (Corrected Version)
"""
Tools for batching MaterialGraph objects for training.
This includes a PyTorch Dataset and a custom collate function.
"""
from typing import List, Tuple, Optional, Dict, Any

import torch
import numpy as np
from torch.utils.data import Dataset

from .struct_to_graph import MaterialGraph, RadiusCutoffGraphConverter, StructureOrMolecule

def collate_fn(batch: List[Tuple[MaterialGraph, ...]]) -> Tuple[MaterialGraph, Tuple[torch.Tensor, ...]]:
    """
    Custom collate function to batch multiple MaterialGraph objects into a single large graph.
    This function is designed to be used with a PyTorch DataLoader.

    Args:
        batch (List[Tuple[MaterialGraph, ...]]): A list where each element is a tuple.
            The first item of the tuple is a MaterialGraph object, and the rest are
            tensors (e.g., energy, forces, stress).

    Returns:
        Tuple[MaterialGraph, Tuple[torch.Tensor, ...]]: A tuple containing:
            - A single batched MaterialGraph.
            - A tuple of batched target tensors.
    """
    graphs, targets_by_type = list(zip(*[(b[0], b[1:]) for b in batch]))

    # Keep track of cumulative number of atoms and bonds
    n_atom_cumsum = torch.cumsum(torch.cat([g.n_atoms for g in graphs]), dim=0)
    n_bond_cumsum = torch.cumsum(torch.cat([g.n_bonds for g in graphs]), dim=0)
    
    # Prepend zero for easier indexing
    n_atom_cumsum_shifted = torch.cat([torch.tensor([0]), n_atom_cumsum[:-1]])
    n_bond_cumsum_shifted = torch.cat([torch.tensor([0]), n_bond_cumsum[:-1]])

    # Batch graph attributes
    batched_attrs = {
        "atom_features": torch.cat([g.atom_features for g in graphs], dim=0),
        "bond_features": torch.cat([g.bond_features for g in graphs], dim=0),
        "atom_positions": torch.cat([g.atom_positions for g in graphs], dim=0),
        "pbc_offsets": torch.cat([g.pbc_offsets for g in graphs], dim=0),
        "n_atoms": torch.cat([g.n_atoms for g in graphs]),
        "n_bonds": torch.cat([g.n_bonds for g in graphs]),
        "lattices": torch.cat([g.lattices for g in graphs if g.lattices is not None], dim=0) if any(g.lattices is not None for g in graphs) else None,
        "state_features": torch.cat([g.state_features for g in graphs if g.state_features is not None], dim=0) if any(g.state_features is not None for g in graphs) else None,
    }

    # Adjust indices for the batched graph
    bond_atom_indices = []
    for i, g in enumerate(graphs):
        bond_atom_indices.append(g.bond_atom_indices + n_atom_cumsum_shifted[i])
    batched_attrs["bond_atom_indices"] = torch.cat(bond_atom_indices, dim=0)

    # Handle three-body information
    graphs_with_3body = [g for g in graphs if g.has_three_body and g.triple_bond_indices is not None]
    if graphs_with_3body:
        batched_attrs["has_three_body"] = True
        batched_attrs["triple_features"] = torch.cat([g.triple_features for g in graphs_with_3body], dim=0)
        batched_attrs["n_triples"] = torch.cat([g.n_triples for g in graphs_with_3body])
        
        triple_bond_indices = []
        graph_indices_with_3body = [i for i, g in enumerate(graphs) if g.has_three_body and g.triple_bond_indices is not None]
        for i, g in zip(graph_indices_with_3body, graphs_with_3body):
            triple_bond_indices.append(g.triple_bond_indices + n_bond_cumsum_shifted[i])
        batched_attrs["triple_bond_indices"] = torch.cat(triple_bond_indices, dim=0)
    else:
        batched_attrs["has_three_body"] = False

    batched_graph = MaterialGraph(**batched_attrs)

    # --- START OF CORRECTED LOGIC FOR TARGETS ---
    batched_targets = []
    if targets_by_type:
        num_targets = len(targets_by_type[0])
        for i in range(num_targets):
            target_tensors = [t[i] for t in targets_by_type if t[i] is not None]
            if not target_tensors:
                batched_targets.append(None)
                continue
            
            # Heuristic to decide whether to stack or concatenate:
            # If the target's first dimension size matches the number of atoms in the graph,
            # it's a per-atom property (like forces) and should be concatenated.
            # Otherwise, it's a per-graph property (like energy or stress) and should be stacked.
            is_per_atom_property = (
                target_tensors[0].ndim > 0 and 
                target_tensors[0].shape[0] == graphs[0].n_atoms.item()
            )

            if is_per_atom_property:
                batched_targets.append(torch.cat(target_tensors, dim=0))
            else:
                batched_targets.append(torch.stack(target_tensors, dim=0))
    # --- END OF CORRECTED LOGIC FOR TARGETS ---
                
    return batched_graph, tuple(batched_targets)


class MaterialGraphDataset(Dataset):
    """
    A PyTorch Dataset for a list of structures and their associated properties.
    """
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