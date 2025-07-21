# m3gnet/graph/batch.py (完整修改后的最终版本)

"""
Tools for batching MaterialGraph objects for training.
"""
from typing import List, Tuple, Dict

import torch
from .struct_to_graph import MaterialGraph

def _batch_graphs(graphs: List[MaterialGraph]) -> MaterialGraph:
    """
    Helper function to batch a list of MaterialGraph objects into a single large graph.
    This function handles the concatenation of attributes and the critical adjustment of indices.
    """
    # Use the device of the first graph's atom_features as the target device
    device = graphs[0].atom_features.device

    # --- Concatenate graph attributes ---
    atom_features = torch.cat([g.atom_features for g in graphs], dim=0)
    
    # Handle potentially empty tensors robustly
    if any(g.atom_positions.numel() > 0 for g in graphs):
        atom_positions = torch.cat([g.atom_positions for g in graphs], dim=0)
    else:
        atom_positions = torch.empty(0, 3, dtype=torch.float32, device=device)

    # NEW: Concatenate bond_distances instead of bond_features
    if any(g.bond_distances.numel() > 0 for g in graphs):
        bond_distances = torch.cat([g.bond_distances for g in graphs], dim=0)
    else:
        bond_distances = torch.empty(0, dtype=torch.float32, device=device)

    if any(g.pbc_offsets.numel() > 0 for g in graphs):
        pbc_offsets = torch.cat([g.pbc_offsets for g in graphs], dim=0)
    else:
        pbc_offsets = torch.empty(0, 3, dtype=torch.float32, device=device)

    # --- Handle index adjustments, which is the core of batching ---
    if any(g.bond_atom_indices.numel() > 0 for g in graphs):
        atom_counts = torch.tensor([g.n_atoms.item() for g in graphs], device=device)
        atom_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device), atom_counts[:-1]]), dim=0)
        
        bond_atom_indices_list = []
        for g, offset in zip(graphs, atom_offsets):
            if g.bond_atom_indices.numel() > 0:
                bond_atom_indices_list.append(g.bond_atom_indices + offset)
        
        bond_atom_indices = torch.cat(bond_atom_indices_list, dim=0)
    else:
        bond_atom_indices = torch.empty(0, 2, dtype=torch.long, device=device)

    # --- Assemble the batched MaterialGraph object ---
    batched_graph = MaterialGraph(
        atom_features=atom_features,
        atom_positions=atom_positions,
        bond_atom_indices=bond_atom_indices,
        bond_distances=bond_distances, #<-- Use new field
        pbc_offsets=pbc_offsets,
        n_atoms=torch.tensor([g.n_atoms.item() for g in graphs], device=device),
        n_bonds=torch.tensor([g.n_bonds.item() for g in graphs], device=device),
        lattices=torch.cat([g.lattices for g in graphs if g.lattices is not None], dim=0) if any(g.lattices is not None for g in graphs) else None,
        state_features=torch.cat([g.state_features for g in graphs if g.state_features is not None], dim=0) if any(g.state_features is not None for g in graphs) else None,
        has_three_body=any(g.has_three_body for g in graphs)
    )

    # --- Batch three-body information if it exists ---
    if batched_graph.has_three_body:
        valid_graphs_3b = [g for g in graphs if g.triple_bond_indices is not None]
        if valid_graphs_3b:
            bond_counts = torch.tensor([g.n_bonds.item() for g in graphs], device=device)
            bond_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device), bond_counts[:-1]]), dim=0)
            
            triple_bond_indices_list = [
                g.triple_bond_indices + offset 
                for g, offset in zip(graphs, bond_offsets) 
                if g.triple_bond_indices is not None
            ]
            if triple_bond_indices_list:
                batched_graph.triple_bond_indices = torch.cat(triple_bond_indices_list, dim=0)
            
            n_triples_list = [g.n_triples.item() for g in graphs if g.n_triples is not None]
            if n_triples_list:
                batched_graph.n_triples = torch.tensor(n_triples_list, device=device)
            
    return batched_graph

def collate_list_of_graphs(batch: List[Tuple[MaterialGraph, ...]]) -> Tuple[MaterialGraph, Tuple[torch.Tensor, ...]]:
    """
    Collates a list of (graph, target1, target2, ...) tuples into a batched graph and batched targets.
    This function is suitable for use as a `collate_fn` in a PyTorch DataLoader.
    """
    if not batch:
        # Create an empty graph structure to handle empty batches gracefully
        return MaterialGraph(
            atom_features=torch.empty(0), bond_atom_indices=torch.empty(0, 2, dtype=torch.long),
            atom_positions=torch.empty(0, 3), pbc_offsets=torch.empty(0, 3),
            n_atoms=torch.empty(0, dtype=torch.long), n_bonds=torch.empty(0, dtype=torch.long),
            bond_distances=torch.empty(0)
        ), tuple()

    graphs = [item[0] for item in batch]
    batched_graph = _batch_graphs(graphs)
    
    # Process targets
    num_targets = len(batch[0]) - 1
    all_targets = []
    if num_targets > 0:
        for i in range(num_targets):
            # Gather all targets for the i-th position
            targets = [item[i + 1] for item in batch]
            stacked_targets = torch.stack(targets, dim=0)
            
            # Ensure the stacked target tensor is always 2D: [batch_size, num_features]
            if stacked_targets.ndim == 1:
                stacked_targets = stacked_targets.view(-1, 1)
            
            all_targets.append(stacked_targets)
        
    return batched_graph, tuple(all_targets)

def collate_potential_graphs(batch: List[Tuple[MaterialGraph, Dict[str, torch.Tensor]]]) -> Tuple[MaterialGraph, Dict[str, torch.Tensor]]:
    """
    Collates a list of (graph, {target_name: target_tensor, ...}) tuples.
    Special handling for 'forces' which are concatenated, while others are stacked.
    """
    graphs = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    batched_graph = _batch_graphs(graphs)
    
    batched_targets = {}
    if not targets_list:
        return batched_graph, batched_targets
        
    keys = targets_list[0].keys()
    for key in keys:
        # Check if any graph has this target
        if not any(key in t for t in targets_list):
            continue

        valid_targets = [t[key] for t in targets_list if key in t and t[key] is not None]
        if not valid_targets:
            continue

        if key == 'forces':
            # Forces from different graphs are concatenated along the atom dimension
            batched_targets[key] = torch.cat(valid_targets, dim=0)
        else:
            # Other targets (like energy, stress) are stacked along the batch dimension
            stacked_targets = torch.stack(valid_targets, dim=0)
            if stacked_targets.ndim == 1:
                batched_targets[key] = stacked_targets.view(-1, 1)
            else:
                batched_targets[key] = stacked_targets
                
    return batched_graph, batched_targets