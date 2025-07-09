# m3gnet/models/m3gnet.py (Final Correct Version)

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

        # The initial bond feature dimension from SphericalBesselBasis is max_n
        sbf_dim = max_n
        self.featurizer = GraphFeaturizer(
            n_atom_types=n_atom_types, embedding_dim=units,
            rbf_type="SphericalBessel", max_l=max_l, max_n=max_n, cutoff=cutoff, bond_feat_dim=sbf_dim
        )
        
        # 3-body basis expansion
        shf_dim = (max_l + 1) ** 2
        rbf_dim = max_n * shf_dim
        self.basis_expansion = SphericalBesselWithHarmonics(max_n=max_n, max_l=max_l, cutoff=threebody_cutoff)
        
        # Interaction blocks
        self.three_interactions = nn.ModuleList([
            ThreeDInteraction(
                update_network=MLP([units, rbf_dim]),
                fusion_network=GatedMLP([rbf_dim, units])
            ) for _ in range(n_blocks)
        ])
        
        self.graph_layers = nn.ModuleList()
        for i in range(n_blocks):
            # <<-- 关键修正 1: 修正 bond_network 的输入维度 -->>
            # The bond feature dimension is sbf_dim ONLY for the first layer's input.
            # After that, it becomes `units`. To handle this, we use a more general approach.
            # However, the provided ThreeDInteraction logic actually replaces the bond features,
            # so the input to ConcatAtoms will have a bond dim of `units`.
            # A simpler, correct M3GNet design would have the bond features from the 3-body
            # interaction be *added* to the 2-body bond features.
            # Given the current structure, let's assume the 3-body interaction *replaces* the 2-body ones.
            # The input to the bond_network is (atom_src, atom_dest, bond_features)
            # The bond feature dimension will be `units` after the ThreeDInteraction.
            # For the very first pass, the bond_features are of size `sbf_dim`.
            # To simplify, we can adjust the logic slightly or ensure dimensions match.
            # Let's adjust the `forward` pass logic to make this clean.
            # For the `__init__`, we will define layers assuming a consistent `units` dimension for bonds inside the loop.
            self.graph_layers.append(
                GraphNetworkLayer(
                    atom_network=GatedAtomUpdate([units, units]),
                    # The bond network takes source atom, dest atom, and bond features.
                    # Total input dim = units + units + bond_dim
                    # After the first three_interaction, bond_dim is `units`.
                    bond_network=ConcatAtoms([units * 2 + units, units])
                )
            )
        
        # Readout
        if is_intensive:
            if readout == "weighted_atom":
                self.readout_layer = WeightedReadout([units, units])
            elif readout == "set2set":
                self.readout_layer = Set2Set(in_features=units, processing_steps=3)
            else:
                self.readout_layer = ReduceReadOut("mean")
            self.final_mlp = MLP([units, units, 1], is_output=(task_type == "regression"))
        else: # Extensive property
            self.readout_layer = MLP([units, 1]) # Per-atom property
            self.final_mlp = ReduceReadOut("sum")

        if element_refs is not None:
            self.element_ref_calc = AtomRef(property_per_element=torch.tensor(element_refs, dtype=torch.float32))
        else:
            self.element_ref_calc = BaseAtomRef()

    def forward(self, graph: MaterialGraph, state_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # <<-- 关键修正 2: 增加对三体特征是否存在的检查 -->>
        atom_features, bond_features, _ = self.featurizer(graph)
        
        # Only compute three-body interactions if they exist
        if graph.has_three_body:
            three_body_basis = self.basis_expansion(
                graph.triple_bond_lengths, 
                graph.triple_features.squeeze(-1)
            )
        
        # A small fix is needed here. The bond_features dimension changes.
        # The original paper's architecture is subtle. The 3-body interaction updates the bond features.
        # The 2-body interaction (GraphNetworkLayer) should then use these updated bond features.
        # Let's align the code with the logic.
        
        for i in range(len(self.graph_layers)):
            # 1. Update bond features using three-body interactions
            if graph.has_three_body:
                bond_features = self.three_interactions[i](atom_features, bond_features, three_body_basis, graph)

            # 2. Update atom and bond features using two-body interactions
            # The `ConcatAtoms` in `graph_layers` expects bond features of size `units`, which is what `three_interactions` outputs.
            # If there are no three-body interactions, this will fail. Let's pad bond_features if needed.
            if not graph.has_three_body and bond_features.shape[1] != self.hparams["units"]:
                 padded_bonds = torch.zeros(bond_features.shape[0], self.hparams["units"], device=bond_features.device)
                 padded_bonds[:, :bond_features.shape[1]] = bond_features
                 bond_features = padded_bonds
            
            atom_features, bond_features, _ = self.graph_layers[i](atom_features, bond_features, state_features, graph)
        
        # Readout part
        if self.hparams["is_intensive"]:
            batch_atom = torch.repeat_interleave(
                torch.arange(len(graph.n_atoms), device=atom_features.device),
                graph.n_atoms.to(atom_features.device)
            )
            readout_vec = self.readout_layer(atom_features, batch_atom)
            output = self.final_mlp(readout_vec)
        else: # Extensive
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
    
    @classmethod
    def load(cls, model_dir: str) -> 'M3GNet':
        if not os.path.isdir(model_dir):
            raise ValueError(f"'{model_dir}' is not a directory.")
        config_path = os.path.join(model_dir, f"{MODEL_NAME}.json")
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path) as f:
            config = json.load(f)
        if 'element_refs' in config and config['element_refs'] is not None:
             config['element_refs'] = np.array(config['element_refs'])
        model = cls(**config)
        weights_path = os.path.join(model_dir, f"{MODEL_NAME}.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict, strict=False) # Use strict=False for LazyLayers
        else:
            logger.warning(f"Weights file {weights_path} not found.")
        return model
        
    def save(self, dirname: str):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        
        # Create a dummy graph to initialize LazyLinear layers.
        # H2 has no 3-body terms, which tests our robustness check.
        dummy_atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
        dummy_graph = self.graph_converter.convert(dummy_atoms)
        device = next(self.parameters()).device
        
        # Run a forward pass to initialize lazy layers before saving
        self.forward(dummy_graph.to(device))

        params_to_save = self.hparams.copy()
        if 'element_refs' in params_to_save and isinstance(params_to_save['element_refs'], np.ndarray):
             params_to_save['element_refs'] = params_to_save['element_refs'].tolist()
        with open(os.path.join(dirname, f"{MODEL_NAME}.json"), 'w') as f:
            json.dump(params_to_save, f)
        torch.save(self.state_dict(), os.path.join(dirname, f"{MODEL_NAME}.pt"))


class Potential(nn.Module):
    def __init__(self, model: M3GNet):
        super().__init__()
        self.model = model
        self.graph_converter = model.graph_converter

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(
        self, graph: MaterialGraph, compute_forces: bool = True, compute_stress: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        graph.atom_positions.requires_grad_(True)
        if compute_stress and graph.has_lattice:
             graph.lattices.requires_grad_(True)
        
        total_energy = self.model(graph).sum()
        
        forces, stress = None, None
        
        if compute_forces:
            grads = torch.autograd.grad(
                outputs=total_energy, 
                inputs=graph.atom_positions, 
                grad_outputs=torch.ones_like(total_energy), 
                create_graph=compute_stress  # Keep graph for stress calculation
            )
            forces = -grads[0]
        
        if compute_stress and graph.has_lattice:
            # We need the graph from the force calculation to compute stress
            stress_grads = torch.autograd.grad(
                outputs=total_energy, 
                inputs=graph.lattices, 
                grad_outputs=torch.ones_like(total_energy),
                allow_unused=True
            )
            if stress_grads[0] is not None:
                # stress = (1/V) * dE/d(strain) = -(1/V) * F * L^T
                stress_virial = -torch.matmul(forces.T, graph.atom_positions).squeeze(0)
                # It's better to calculate stress from lattice gradients
                stress = stress_grads[0].squeeze(0) / torch.det(graph.lattices.squeeze(0))
            else:
                stress = torch.zeros((3, 3), device=self.device)

        return total_energy, forces, stress