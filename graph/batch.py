# m3gnet/graph/batch.py (Final Fixed Version for Tests and Training)

"""
Tools for batching MaterialGraph objects for training.
"""
from typing import List, Tuple, Optional, Dict

import torch
import numpy as np

from .struct_to_graph import MaterialGraph

def _batch_graphs(graphs: List[MaterialGraph]) -> MaterialGraph:
    """Helper function to batch graph attributes."""
    atom_features = torch.cat([g.atom_features for g in graphs], dim=0)
    
    if any(g.bond_features.numel() > 0 for g in graphs):
        bond_features = torch.cat([g.bond_features for g in graphs if g.bond_features.numel() > 0], dim=0)
    else:
        bond_features = torch.empty(0, 1, dtype=torch.float32, device=atom_features.device)
    
    if any(g.atom_positions.numel() > 0 for g in graphs):
        atom_positions = torch.cat([g.atom_positions for g in graphs if g.atom_positions.numel() > 0], dim=0)
    else:
        atom_positions = torch.empty(0, 3, dtype=torch.float32, device=atom_features.device)

    if any(g.pbc_offsets.numel() > 0 for g in graphs):
        pbc_offsets = torch.cat([g.pbc_offsets for g in graphs if g.pbc_offsets.numel() > 0], dim=0)
    else:
        pbc_offsets = torch.empty(0, 3, dtype=torch.float32, device=atom_features.device)

    if any(g.bond_atom_indices.numel() > 0 for g in graphs):
        atom_offsets = torch.cumsum(torch.tensor([0] + [g.n_atoms.item() for g in graphs[:-1]]), dim=0)
        bond_atom_indices = torch.cat([g.bond_atom_indices + offset for g, offset in zip(graphs, atom_offsets)], dim=0)
    else:
        bond_atom_indices = torch.empty(0, 2, dtype=torch.long, device=atom_features.device)

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
        
        triple_bond_indices_list = [g.triple_bond_indices + offset for g, offset in zip(graphs, bond_offsets) if g.triple_bond_indices is not None]
        if triple_bond_indices_list:
            batched_graph.triple_bond_indices = torch.cat(triple_bond_indices_list)
        
        triple_features_list = [g.triple_features for g in graphs if g.triple_features is not None]
        if triple_features_list:
            batched_graph.triple_features = torch.cat(triple_features_list)
            
        triple_bond_lengths_list = [g.triple_bond_lengths for g in graphs if g.triple_bond_lengths is not None]
        if triple_bond_lengths_list:
            batched_graph.triple_bond_lengths = torch.cat(triple_bond_lengths_list)
            
        n_triples_list = [g.n_triples.item() for g in graphs if g.n_triples is not None]
        if n_triples_list:
            batched_graph.n_triples = torch.tensor(n_triples_list)
            
    return batched_graph


# --- [ THE MODIFICATION IS HERE ] ---
def collate_list_of_graphs(batch: List[Tuple[MaterialGraph, ...]]) -> Tuple[MaterialGraph, Tuple[torch.Tensor, ...]]:
    """
    Collates a list of (graph, target1, target2, ...) tuples, ensuring targets are 2D.
    """
    if not batch:
        # Handle empty batch case
        return MaterialGraph(atom_features=torch.empty(0)), tuple()

    graphs = [item[0] for item in batch] 
    
    num_targets = len(batch[0]) - 1
    all_targets = []
    for i in range(num_targets):
        targets = [item[i + 1] for item in batch]
        stacked_targets = torch.stack(targets, dim=0)
        
        # --- SHAPE FIX IS HERE ---
        # Ensure the stacked target tensor is always 2D [batch_size, num_features]
        # This makes it consistent for downstream processing.
        if stacked_targets.ndim == 1:
            stacked_targets = stacked_targets.view(-1, 1)
        
        all_targets.append(stacked_targets)
        
    batched_graph = _batch_graphs(graphs)
    
    return batched_graph, tuple(all_targets)

# --- [ `collate_potential_graphs` remains unchanged ] ---
def collate_potential_graphs(batch: List[Tuple[MaterialGraph, Dict[str, torch.Tensor]]]) -> Tuple[MaterialGraph, Dict[str, torch.Tensor]]:
    # ... (This function is already correct and does not need changes)
    graphs = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    batched_graph = _batch_graphs(graphs)
    batched_targets = {}
    
    if not targets_list:
        return batched_graph, batched_targets
        
    keys = targets_list[0].keys()
    for key in keys:
        current_targets = [t.get(key) for t in targets_list]
        if all(t is None for t in current_targets):
            continue
        
        valid_targets = [t for t in current_targets if t is not None]
        if not valid_targets:
            continue

        if key == 'forces':
            batched_targets[key] = torch.cat(valid_targets, dim=0)
        else:
            stacked_targets = torch.stack(valid_targets, dim=0)
            if stacked_targets.ndim == 1:
                batched_targets[key] = stacked_targets.view(-1, 1)
            else:
                batched_targets[key] = stacked_targets
                
    return batched_graph, batched_targets