# m3gnet/layers/_graph_layers.py (Final Fixed Version)

"""Core graph convolutional layers."""

import torch
import torch.nn as nn
from torch_scatter import scatter
from typing import List

from ._core import AtomEmbedding, GatedMLP, MLP
from ._basis import SphericalBesselWithHarmonics, GaussianBasis, SphericalBesselBasis

# ... (BaseAtomRef, AtomRef, ReduceState, ConcatAtoms are unchanged) ...
class BaseAtomRef(nn.Module):
    def forward(self, graph) -> torch.Tensor:
        batch_size = len(graph.n_atoms)
        return torch.zeros(batch_size, 1, device=graph.atom_features.device)

class AtomRef(nn.Module):
    def __init__(self, property_per_element: torch.Tensor):
        super().__init__()
        self.register_buffer("property_per_element", property_per_element)

    def forward(self, graph) -> torch.Tensor:
        atom_props = self.property_per_element[graph.atom_features.long().squeeze()]
        batch_atom = torch.repeat_interleave(
            torch.arange(len(graph.n_atoms), device=atom_props.device),
            graph.n_atoms.to(atom_props.device)
        )
        return scatter(atom_props, batch_atom, dim=0, reduce="sum").view(-1, 1)

class ReduceState(nn.Module):
    def __init__(self, reducer: str = "mean"):
        super().__init__()
        self.reducer = reducer

    def forward(self, features: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return scatter(features, index, dim=0, reduce=self.reducer)

class ConcatAtoms(nn.Module):
    def __init__(self, neurons: List[int]):
        super().__init__()
        self.mlp = GatedMLP(neurons)

    def forward(self, atom_features, bond_features, graph) -> torch.Tensor:
        sender_atoms = atom_features[graph.bond_atom_indices[:, 0]]
        receiver_atoms = atom_features[graph.bond_atom_indices[:, 1]]
        concatenated = torch.cat([sender_atoms, receiver_atoms, bond_features], dim=-1)
        return self.mlp(concatenated)

class GatedAtomUpdate(nn.Module):
    def __init__(self, neurons: List[int]):
        super().__init__()
        self.mlp = GatedMLP(neurons)
    
    def forward(self, atom_features, bond_features, graph) -> torch.Tensor:
        messages = self.mlp(bond_features)
        summed_messages = scatter(messages, graph.bond_atom_indices[:, 1], dim=0, reduce="sum")
        return atom_features + summed_messages

# --- Corrected ThreeDInteraction ---
class ThreeDInteraction(nn.Module):
    """Three-body interaction layer."""
    def __init__(self, update_network: nn.Module, fusion_network: nn.Module):
        """
        Args:
            update_network (nn.Module): Network to process third-atom features.
            fusion_network (nn.Module): Network to fuse 3-body messages into bond features.
        """
        super().__init__()
        self.update_network = update_network
        self.fusion_network = fusion_network
    
    def forward(self, atom_features, bond_features, three_body_basis, graph) -> torch.Tensor:
        third_atom_indices = graph.bond_atom_indices[graph.triple_bond_indices[:, 1], 1]
        third_atom_features = self.update_network(atom_features[third_atom_indices])
        
        messages = three_body_basis * third_atom_features
        
        center_bond_indices = graph.triple_bond_indices[:, 0]
        summed_messages = scatter(messages, center_bond_indices, dim=0, reduce="sum", out=torch.zeros(bond_features.shape[0], messages.shape[1], device=bond_features.device))
        
        # Fuse with original bond features
        return bond_features + self.fusion_network(summed_messages)

# ... (GraphNetworkLayer, GraphFeaturizer are unchanged) ...
class GraphNetworkLayer(nn.Module):
    def __init__(self, atom_network, bond_network, state_network=None):
        super().__init__()
        self.atom_network = atom_network
        self.bond_network = bond_network
        self.state_network = state_network
        
    def forward(self, atom_features, bond_features, state_features, graph):
        if self.bond_network:
            bond_features = self.bond_network(atom_features, bond_features, graph)
        if self.atom_network:
            atom_features = self.atom_network(atom_features, bond_features, graph)
        if self.state_network:
            state_features = self.state_network(atom_features, bond_features, state_features, graph)
        return atom_features, bond_features, state_features

class GraphFeaturizer(nn.Module):
    def __init__(self, n_atom_types, embedding_dim, rbf_type="Gaussian", **rbf_kwargs):
        super().__init__()
        self.atom_embedding = AtomEmbedding(n_atom_types, embedding_dim)
        if rbf_type == "Gaussian":
            self.bond_expansion = GaussianBasis(**rbf_kwargs)
        else: # SphericalBessel
            self.bond_expansion = SphericalBesselBasis(**rbf_kwargs)
    
    def forward(self, graph):
        atom_features = self.atom_embedding(graph.atom_features)
        bond_features = self.bond_expansion(graph.bond_features)
        return atom_features, bond_features, graph.state_features