# m3gnet/models/m3gnet.py (Final Modified Version after Layer Changes)

import json
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from ase import Atoms
from pymatgen.core import Molecule, Structure

from m3gnet.graph import MaterialGraph, RadiusCutoffGraphConverter
from m3gnet.layers import (
    AtomRef, BaseAtomRef, ConcatAtoms, GatedAtomUpdate,
    GatedMLP, GraphFeaturizer, GraphNetworkLayer, MLP, ReduceReadOut,
    Set2Set, SphericalBesselWithHarmonics, ThreeDInteraction, WeightedReadout
)

logger = logging.getLogger(__name__)
MODEL_NAME = "m3gnet"

class M3GNet(nn.Module):
    """
    The main M3GNet model for energy prediction. This class orchestrates the
    graph conversion, feature extraction, and interaction blocks.
    """
    def __init__(
        self, max_n: int = 3, max_l: int = 3, n_blocks: int = 3, units: int = 64,
        cutoff: float = 5.0, threebody_cutoff: float = 4.0, n_atom_types: int = 95, 
        is_intensive: bool = False, readout: str = "set2set", 
        task_type: str = "regression", embedding_type: str = "attention",
        mean: float = 0.0, std: float = 1.0, element_refs: Optional[np.ndarray] = None, **kwargs
    ):
        super().__init__()
        self.hparams = {
            "max_n": max_n, "max_l": max_l, "n_blocks": n_blocks, "units": units,
            "cutoff": cutoff, "threebody_cutoff": threebody_cutoff, "n_atom_types": n_atom_types,
            "is_intensive": is_intensive, "readout": readout, "task_type": task_type,
            "embedding_type": embedding_type, "mean": mean, "std": std, 
            "element_refs": element_refs, **kwargs
        }
        
        self.graph_converter = RadiusCutoffGraphConverter(cutoff=cutoff, threebody_cutoff=threebody_cutoff)
        
        # <<<<<<<<<<<<<<<<< UPDATED INITIALIZATION <<<<<<<<<<<<<<<<<
        # GraphFeaturizer no longer takes RBF arguments
        self.featurizer = GraphFeaturizer(
            n_atom_types=n_atom_types, embedding_dim=units, 
            embedding_type=embedding_type
        )
        
        self.bond_projection = MLP([1, units]) # Input is bond length (1 dim)
        self.basis_expansion = SphericalBesselWithHarmonics(max_n=max_n, max_l=max_l, cutoff=cutoff)
        
        shf_dim = (max_l + 1) ** 2
        rbf_dim = max_n * shf_dim
        self.three_interactions = nn.ModuleList([
            ThreeDInteraction(update_network=MLP([units, rbf_dim]), fusion_network=GatedMLP([rbf_dim, units])) 
            for _ in range(n_blocks)
        ])
        self.graph_layers = nn.ModuleList([
            GraphNetworkLayer(atom_network=GatedAtomUpdate([units, units]), bond_network=ConcatAtoms([units * 2 + units, units])) 
            for _ in range(n_blocks)
        ])

        if is_intensive:
            if readout == "set2set":
                self.readout_layer = Set2Set(in_features=units, processing_steps=3)
                self.final_mlp = MLP([units * 2, units, 1], is_output=(task_type == "regression"))
            elif readout == "weighted_atom":
                self.readout_layer = WeightedReadout([units, units])
                self.final_mlp = MLP([units, units, 1], is_output=(task_type == "regression"))
            else: # "mean"
                self.readout_layer = ReduceReadOut("mean")
                self.final_mlp = MLP([units, units, 1], is_output=(task_type == "regression"))
        else: # Extensive property
            self.readout_layer = MLP([units, 1])
            self.final_mlp = ReduceReadOut("sum")
            
        self.element_ref_calc = AtomRef(property_per_element=torch.tensor(element_refs, dtype=torch.float32)) if element_refs is not None else BaseAtomRef()

    def _compute_geometries(self, graph: MaterialGraph) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        A helper method to compute bond and three-body geometries from atom positions.
        """
        sender_indices = graph.bond_atom_indices[:, 0]
        receiver_indices = graph.bond_atom_indices[:, 1]
        
        sender_pos = graph.atom_positions[sender_indices]
        receiver_pos = graph.atom_positions[receiver_indices]
        
        vec_ij = receiver_pos - sender_pos
        
        has_lattice = graph.lattices is not None
        if has_lattice and graph.pbc_offsets is not None and graph.pbc_offsets.numel() > 0:
            num_bonds_per_graph = graph.n_bonds.to(graph.lattices.device)
            repeated_lattices = torch.repeat_interleave(graph.lattices, num_bonds_per_graph, dim=0)
            vec_ij += torch.einsum('bi,bij->bj', graph.pbc_offsets, repeated_lattices)
        
        bond_lengths = torch.norm(vec_ij, dim=1, keepdim=True)

        triple_bond_lengths = None
        cos_theta = None
        if graph.has_three_body and graph.triple_bond_indices is not None:
            bond_indices_1 = graph.triple_bond_indices[:, 0]
            bond_indices_2 = graph.triple_bond_indices[:, 1]
            
            vec1 = vec_ij[bond_indices_1]
            vec2 = vec_ij[bond_indices_2]
            
            triple_bond_lengths = torch.norm(vec2, dim=1)
            
            vec1_norm = torch.norm(vec1, dim=1, keepdim=True)
            vec2_norm = torch.norm(vec2, dim=1, keepdim=True)
            normalized_vec1 = vec1 / (vec1_norm + 1e-8)
            normalized_vec2 = vec2 / (vec2_norm + 1e-8)

            cos_theta = torch.sum(normalized_vec1 * normalized_vec2, dim=1)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            
        return bond_lengths, triple_bond_lengths, cos_theta

    def forward(self, graph: MaterialGraph, state_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The forward pass of the M3GNet model.
        """
        # Step 1: Dynamically compute bond and angle geometries
        bond_lengths, triple_bond_lengths, cos_theta = self._compute_geometries(graph)

        # <<<<<<<<<<<<<<<<< UPDATED FORWARD PASS <<<<<<<<<<<<<<<<<
        # Step 2: Featurize atoms. The featurizer now only returns atom and state features.
        atom_features, graph_state_features = self.featurizer(graph)
        if state_features is None:
            state_features = graph_state_features

        # Project bond lengths into feature space
        bond_features = self.bond_projection(bond_lengths)
        
        # Step 3: Expand three-body interactions into a basis set
        if graph.has_three_body and triple_bond_lengths is not None and cos_theta is not None:
            three_body_basis = self.basis_expansion(triple_bond_lengths, cos_theta.unsqueeze(-1))
        else:
            three_body_basis = None
            
        # Step 4: Main interaction loop
        for i in range(self.hparams["n_blocks"]):
            if three_body_basis is not None:
                bond_features = self.three_interactions[i](atom_features, bond_features, three_body_basis, graph)
            atom_features, bond_features, _ = self.graph_layers[i](atom_features, bond_features, state_features, graph)
        
        # Step 5: Readout and final MLP
        batch_atom = torch.repeat_interleave(
            torch.arange(len(graph.n_atoms), device=atom_features.device), 
            graph.n_atoms.to(atom_features.device)
        )
        if self.hparams["is_intensive"]:
            readout_vec = self.readout_layer(atom_features, batch_atom)
            output = self.final_mlp(readout_vec)
        else:
            per_atom_output = self.readout_layer(atom_features)
            output = self.final_mlp(per_atom_output, batch_atom)
        
        # Step 6: Apply scaling and element reference
        output = output * self.hparams["std"] + self.hparams["mean"]
        output += self.element_ref_calc(graph)
        
        return output
    
    @classmethod
    def load(cls, model_dir: str) -> 'M3GNet':
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
        # ... (This method should work without changes)
        if not os.path.isdir(dirname): os.makedirs(dirname)
        
        params_to_save = self.hparams.copy()
        if 'element_refs' in params_to_save and isinstance(params_to_save['element_refs'], np.ndarray):
             params_to_save['element_refs'] = params_to_save['element_refs'].tolist()
        with open(os.path.join(dirname, f"{MODEL_NAME}.json"), 'w') as f:
            json.dump(params_to_save, f, indent=2)

        torch.save(self.state_dict(), os.path.join(dirname, f"{MODEL_NAME}.pt"))


class Potential(nn.Module):
    """
    A wrapper class for M3GNet that computes energy, forces, and stress.
    Forces and stress are calculated via auto-differentiation of energy.
    """
    def __init__(self, model: M3GNet):
        super().__init__()
        self.model = model
        self.graph_converter = model.graph_converter
        
    @property
    def device(self):
        """Returns the device of the model."""
        return next(self.model.parameters()).device
        
    def forward(
        self, graph: MaterialGraph, compute_forces: bool = True, compute_stress: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        with torch.enable_grad():
            graph.atom_positions.requires_grad_(True)
            has_lattice = graph.lattices is not None
            if compute_stress and has_lattice:
                graph.lattices.requires_grad_(True)
            
            total_energy = self.model(graph)
            energy_sum_for_grad = total_energy.sum()

            if compute_forces:
                # V V V V V FIX IS HERE V V V V V
                # We need retain_graph=True here to allow the main training loop's 
                # loss.backward() to function after forces are computed.
                grads = torch.autograd.grad(
                    outputs=energy_sum_for_grad, 
                    inputs=graph.atom_positions, 
                    create_graph=True, # Allows for gradient of gradient (e.g., Hessian)
                    retain_graph=True, # Keeps the graph alive for a second backward pass
                )
                forces = -grads[0]
            else:
                forces = None
            
            if compute_stress and has_lattice:
                # Since forces already retained the graph, this grad call will work.
                # If forces are not computed, this call might need retain_graph too.
                # For safety, we can add it here as well.
                stress_grads = torch.autograd.grad(
                    outputs=energy_sum_for_grad, 
                    inputs=graph.lattices, 
                    allow_unused=True,
                    retain_graph=True # Add for safety, in case forces are not computed
                )
                if stress_grads[0] is not None:
                    volume = torch.det(graph.lattices).view(-1, 1, 1)
                    stress = stress_grads[0] / volume
                else: 
                    stress = torch.zeros((len(graph.lattices), 3, 3), device=self.device)
            else:
                stress = None

        # Format the output for single vs. multiple graphs
        is_single_graph = graph.n_atoms.size(0) == 1
        if is_single_graph:
            squeezed_energy = total_energy.squeeze(0)
            squeezed_stress = stress.squeeze(0) if stress is not None else None
            return squeezed_energy, forces, squeezed_stress
        else:
            return total_energy, forces, stress