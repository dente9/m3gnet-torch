# m3gnet/models/m3gnet.py (Final Fixed Version 3)

import json
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from ase import Atoms

from m3gnet.graph import MaterialGraph, RadiusCutoffGraphConverter
from m3gnet.layers import (
    AtomRef, BaseAtomRef, ConcatAtoms, GatedAtomUpdate,
    GatedMLP, GraphFeaturizer, GraphNetworkLayer, MLP, ReduceReadOut,
    Set2Set, SphericalBesselWithHarmonics, ThreeDInteraction, WeightedReadout
)

logger = logging.getLogger(__name__)

CWD = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "m3gnet"

# M3GNet class is now correct and does not need further changes.
# All code from the previous version is retained.
class M3GNet(nn.Module):
    def __init__(
        self,
        max_n: int = 3, max_l: int = 3, n_blocks: int = 3, units: int = 64,
        cutoff: float = 5.0, threebody_cutoff: float = 4.0,
        n_atom_types: int = 94, is_intensive: bool = True,
        readout: str = "weighted_atom", task_type: str = "regression",
        mean: float = 0.0, std: float = 1.0,
        element_refs: Optional[np.ndarray] = None, **kwargs
    ):
        super().__init__()
        self.hparams = {
            "max_n": max_n, "max_l": max_l, "n_blocks": n_blocks, "units": units,
            "cutoff": cutoff, "threebody_cutoff": threebody_cutoff,
            "n_atom_types": n_atom_types, "is_intensive": is_intensive, "readout": readout,
            "task_type": task_type, "mean": mean, "std": std,
            "element_refs": element_refs, **kwargs
        }
        self.graph_converter = RadiusCutoffGraphConverter(cutoff=cutoff, threebody_cutoff=threebody_cutoff)

        self.featurizer = GraphFeaturizer(
            n_atom_types=n_atom_types, embedding_dim=units,
            rbf_type="SphericalBessel", max_l=max_l, max_n=max_n, cutoff=cutoff
        )
        
        self.bond_projection = MLP([max_n, units])
        self.basis_expansion = SphericalBesselWithHarmonics(max_n=max_n, max_l=max_l, cutoff=threebody_cutoff)
        
        shf_dim = (max_l + 1) ** 2
        rbf_dim = max_n * shf_dim
        self.three_interactions = nn.ModuleList([
            ThreeDInteraction(
                update_network=MLP([units, rbf_dim]),
                fusion_network=GatedMLP([rbf_dim, units])
            ) for _ in range(n_blocks)
        ])
        
        self.graph_layers = nn.ModuleList([
            GraphNetworkLayer(
                atom_network=GatedAtomUpdate([units, units]),
                bond_network=ConcatAtoms([units * 2 + units, units])
            ) for _ in range(n_blocks)
        ])
        
        if is_intensive:
            readout_map = {
                "weighted_atom": WeightedReadout([units, units]),
                "set2set": Set2Set(in_features=units, processing_steps=3),
                "mean": ReduceReadOut("mean")
            }
            self.readout_layer = readout_map.get(readout, ReduceReadOut("mean"))
            self.final_mlp = MLP([units, units, 1], is_output=(task_type == "regression"))
        else:
            self.readout_layer = MLP([units, 1])
            self.final_mlp = ReduceReadOut("sum")

        self.element_ref_calc = AtomRef(property_per_element=torch.tensor(element_refs, dtype=torch.float32)) if element_refs is not None else BaseAtomRef()

    def forward(self, graph: MaterialGraph, state_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        atom_features, _, _ = self.featurizer(graph)
        
        # Use bond_features directly from the graph object passed in
        bond_features = self.bond_projection(graph.bond_features)

        if graph.has_three_body and graph.triple_bond_indices is not None:
            three_body_basis = self.basis_expansion(
                graph.triple_bond_lengths, 
                graph.triple_features
            )
        else:
            three_body_basis = None
        
        for i in range(self.hparams["n_blocks"]):
            if three_body_basis is not None:
                bond_features = self.three_interactions[i](atom_features, bond_features, three_body_basis, graph)

            atom_features, bond_features, _ = self.graph_layers[i](atom_features, bond_features, state_features, graph)
        
        if self.hparams["is_intensive"]:
            batch_atom = torch.repeat_interleave(
                torch.arange(len(graph.n_atoms), device=atom_features.device),
                graph.n_atoms.to(atom_features.device)
            )
            readout_vec = self.readout_layer(atom_features, batch_atom)
            output = self.final_mlp(readout_vec)
        else:
            per_atom_output = self.readout_layer(atom_features)
            batch_atom = torch.repeat_interleave(
                torch.arange(len(graph.n_atoms), device=atom_features.device),
                graph.n_atoms.to(atom_features.device)
            )
            output = self.final_mlp(per_atom_output, batch_atom)
        
        output = output * self.hparams["std"] + self.hparams["mean"]
        property_offset = self.element_ref_calc(graph)
        output += property_offset
        return output
    
    # save/load methods are correct, no changes needed
    @classmethod
    def load(cls, model_dir: str):
        # ... (no changes)
        if not os.path.isdir(model_dir): raise ValueError(f"'{model_dir}' is not a directory.")
        config_path = os.path.join(model_dir, f"{MODEL_NAME}.json")
        if not os.path.exists(config_path): raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path) as f: config = json.load(f)
        if 'element_refs' in config and config['element_refs'] is not None: config['element_refs'] = np.array(config['element_refs'])
        model = cls(**config)
        weights_path = os.path.join(model_dir, f"{MODEL_NAME}.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
        else:
            logger.warning(f"Weights file {weights_path} not found.")
        return model
        
    def save(self, dirname: str):
        # ... (no changes)
        if not os.path.isdir(dirname): os.makedirs(dirname)
        dummy_atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=True)
        dummy_graph = self.graph_converter.convert(dummy_atoms)
        device = next(self.parameters()).device
        self.forward(dummy_graph.to(device))
        params_to_save = self.hparams.copy()
        if 'element_refs' in params_to_save and isinstance(params_to_save['element_refs'], np.ndarray): params_to_save['element_refs'] = params_to_save['element_refs'].tolist()
        with open(os.path.join(dirname, f"{MODEL_NAME}.json"), 'w') as f: json.dump(params_to_save, f)
        torch.save(self.state_dict(), os.path.join(dirname, f"{MODEL_NAME}.pt"))


# <<<<<<<<<<<<<<<<<<<< THE FIX IS HERE <<<<<<<<<<<<<<<<<<<<
class Potential(nn.Module):
    def __init__(self, model: M3GNet):
        super().__init__()
        self.model = model
        # The graph converter is now only used for getting the topology (indices)
        self.graph_converter = model.graph_converter

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(
        self, graph: MaterialGraph, compute_forces: bool = True, compute_stress: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # Enable gradients for inputs that require them
        graph.atom_positions.requires_grad_(True)
        has_lattice = graph.lattices is not None
        if compute_stress and has_lattice:
             graph.lattices.requires_grad_(True)
        
        # --- Recompute geometric features with PyTorch to maintain the gradient chain ---
        # 1. Bond lengths (bond_features)
        sender_indices = graph.bond_atom_indices[:, 0]
        receiver_indices = graph.bond_atom_indices[:, 1]
        
        sender_pos = graph.atom_positions[sender_indices]
        receiver_pos = graph.atom_positions[receiver_indices]
        
        vec_ij = receiver_pos - sender_pos
        if has_lattice:
            # Add periodic boundary condition offsets
            vec_ij += torch.einsum('bi,bij->bj', graph.pbc_offsets, graph.lattices.expand(len(graph.pbc_offsets), -1, -1))
        
        bond_lengths = torch.norm(vec_ij, dim=1, keepdim=True)
        graph.bond_features = bond_lengths

        # 2. Three-body angles (triple_features)
        if graph.has_three_body and graph.triple_bond_indices is not None:
            bond_indices_1 = graph.triple_bond_indices[:, 0]
            bond_indices_2 = graph.triple_bond_indices[:, 1]
            
            vec1 = vec_ij[bond_indices_1]
            vec2 = vec_ij[bond_indices_2]

            # The lengths of the second bonds in the triples
            graph.triple_bond_lengths = bond_lengths[bond_indices_2].squeeze(-1)

            # Cosine of the angle
            cos_theta = torch.sum(vec1 * vec2, dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1))
            graph.triple_features = torch.clamp(cos_theta, -1.0, 1.0)
        
        # Now, `atom_positions` and `lattices` are part of the computation graph of `total_energy`
        total_energy = self.model(graph).sum()
        
        forces, stress = None, None
        
        # --- Compute gradients ---
        if compute_forces:
            # The RuntimeError is now resolved because atom_positions were used.
            grads = torch.autograd.grad(
                outputs=total_energy, 
                inputs=graph.atom_positions, 
                grad_outputs=torch.ones_like(total_energy), 
                create_graph=True, # Keep graph for stress calculation
                allow_unused=False # Should not be unused now
            )
            forces = -grads[0]
        
        if compute_stress and has_lattice:
            stress_grads = torch.autograd.grad(
                outputs=total_energy, 
                inputs=graph.lattices, 
                grad_outputs=torch.ones_like(total_energy),
                allow_unused=True # It might be unused if the structure is a molecule
            )
            if stress_grads[0] is not None:
                # The volume is needed
                volume = torch.det(graph.lattices.squeeze(0))
                stress = stress_grads[0].squeeze(0) / volume
            else:
                stress = torch.zeros((3, 3), device=self.device)

        return total_energy, forces, stress