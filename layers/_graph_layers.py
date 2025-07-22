# m3gnet/layers/_graph_layers.py (Final Modified Version)

"""Core graph convolutional layers."""
import torch
import torch.nn as nn
from torch_scatter import scatter
from typing import List, Optional,Tuple

# Import our new embedding layer
from ._core import AtomEmbedding, GatedMLP, MLP, AttentionAtomEmbedding
from ._basis import SphericalBesselWithHarmonics, GaussianBasis, SphericalBesselBasis

class BaseAtomRef(nn.Module):
    
    def forward(self, graph) -> torch.Tensor:
        batch_size = len(graph.n_atoms)
        return torch.zeros(batch_size, 1, device=graph.atom_features.device)

class AtomRef(nn.Module):
    
    def __init__(self, property_per_element: torch.Tensor):
        super().__init__()
        self.register_buffer("property_per_element", property_per_element)
    def forward(self, graph) -> torch.Tensor:
        # Use .squeeze() in case atom_features is [N, 1]
        atom_indices = graph.atom_features.long().squeeze(-1)
        atom_props = self.property_per_element[atom_indices]
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
        total_num_atoms = atom_features.size(0)
        summed_messages = scatter(
            messages, graph.bond_atom_indices[:, 1], dim=0, 
            dim_size=total_num_atoms, reduce="sum"
        )
        return atom_features + summed_messages


class ThreeDInteraction(nn.Module):
    
    def __init__(self, update_network: nn.Module, fusion_network: nn.Module):
        super().__init__()
        self.update_network = update_network
        self.fusion_network = fusion_network
    def forward(self, atom_features, bond_features, three_body_basis, graph) -> torch.Tensor:
        third_atom_indices = graph.bond_atom_indices[graph.triple_bond_indices[:, 1], 1]
        third_atom_features = self.update_network(atom_features[third_atom_indices])
        messages = three_body_basis * third_atom_features
        center_bond_indices = graph.triple_bond_indices[:, 0]
        total_num_bonds = bond_features.size(0)
        # Use out= argument for in-place scatter to avoid creating new zero tensors on the fly
        summed_messages = scatter(
            messages, center_bond_indices, dim=0, reduce="sum", 
            out=torch.zeros(total_num_bonds, messages.shape[1], device=bond_features.device)
        )
        return bond_features + self.fusion_network(summed_messages)

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

# <<<<<<<<<<<<<<<<< CORE MODIFICATION HERE <<<<<<<<<<<<<<<<<
class GraphFeaturizer(nn.Module):
    """
    Handles the initial featurization of atoms in the graph, converting atomic
    numbers into feature vectors. It no longer handles bond features.
    """
    def __init__(
        self, 
        n_atom_types: int, 
        embedding_dim: int, 
        embedding_type: str = "attention", # New parameter to choose embedding type
    ):
        super().__init__()
        
        self.embedding_type = embedding_type.lower()
        if self.embedding_type == "attention":
            self.atom_embedding = AttentionAtomEmbedding(n_atom_types, embedding_dim)
            print("Using Context-Aware Attention Atom Embedding.")
        else: # Default to standard embedding
            self.atom_embedding = AtomEmbedding(n_atom_types, embedding_dim)
            print("Using Standard Atom Embedding.")

        # The bond_expansion is REMOVED because bond features are now calculated
        # dynamically inside the main M3GNet model from bond lengths.
        # self.bond_expansion = ...
    
    def forward(self, graph: "MaterialGraph") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Takes a graph and returns initial atom features and state features.
        """
        if self.embedding_type == "attention":
            # The AttentionAtomEmbedding's forward pass expects the whole graph.
            atom_features = self.atom_embedding(graph)
        else: 
            # The standard AtomEmbedding just needs the atom numbers.
            atom_features = self.atom_embedding(graph.atom_features)
            
        # We no longer process or return bond features from the featurizer.
        # The model will generate them dynamically.
        # bond_features = self.bond_expansion(graph.bond_features) 
        
        # It now returns atom_features and state_features
        return atom_features, graph.state_features