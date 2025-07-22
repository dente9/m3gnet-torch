# m3gnet/layers/_core.py (Final Version with Attention Embedding)

"""Core neural network building blocks."""

from typing import List, Optional, Callable

import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_softmax
class MLP(nn.Module):
    """Multi-layer perceptron using LazyLinear for deferred initialization."""
    def __init__(
        self,
        neurons: List[int],
        activation: Optional[Callable] = nn.SiLU(),
        is_output: bool = False,
    ):
        super().__init__()
        self.neurons = neurons
        self.activation = activation
        self.is_output = is_output

        self.layers = nn.ModuleList()
        # Use LazyLinear for the first layer to infer input size
        self.layers.append(nn.LazyLinear(self.neurons[0]))
        if not (self.is_output and len(self.neurons) == 1):
            self.layers.append(self.activation)

        for i in range(1, len(self.neurons)):
            self.layers.append(nn.Linear(self.neurons[i-1], self.neurons[i]))
            if not (self.is_output and i == len(self.neurons) - 1):
                self.layers.append(self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class GatedMLP(nn.Module):
    """Gated multi-layer perceptron."""
    def __init__(self, neurons: List[int], activation: Optional[Callable] = nn.SiLU()):
        super().__init__()
        self.linear_mlp = MLP(neurons, activation)
        self.gating_mlp = MLP(neurons, activation, is_output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_mlp(x) * torch.sigmoid(self.gating_mlp(x))

class AtomEmbedding(nn.Module):
    """Standard, context-free atom embedding layer."""
    def __init__(self, n_atom_types: int, embedding_dim: int, trainable: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, embedding_dim)
        self.embedding.weight.requires_grad = trainable

    def forward(self, atom_numbers: torch.Tensor) -> torch.Tensor:
        return self.embedding(atom_numbers.long().squeeze(-1))

# <<<<<<<<<<<<<<<<<<<< NEW ADVANCED EMBEDDING LAYER <<<<<<<<<<<<<<<<<<<<
class AttentionAtomEmbedding(nn.Module):
    """
    Context-aware atom embedding using a self-attention mechanism.
    Each atom's initial embedding is refined by attending to its local neighborhood.
    """
    def __init__(self, n_atom_types: int, embedding_dim: int, trainable_base_embedding: bool = True):
        super().__init__()
        self.base_embedding = nn.Embedding(n_atom_types, embedding_dim)
        self.base_embedding.weight.requires_grad = trainable_base_embedding

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.final_mlp = MLP([embedding_dim, embedding_dim])
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, graph) -> torch.Tensor:
        """
        The forward pass now requires the whole graph to access neighborhood info.
        """
        base_atom_embeds = self.base_embedding(graph.atom_features.long().squeeze(-1))
        
        queries = self.q_proj(base_atom_embeds)
        keys = self.k_proj(base_atom_embeds)
        values = self.v_proj(base_atom_embeds)
        
        neighbor_indices = graph.bond_atom_indices[:, 0]
        center_indices = graph.bond_atom_indices[:, 1]
        
        neighbor_keys = keys[neighbor_indices]
        neighbor_values = values[neighbor_indices]
        center_queries = queries[center_indices]
        
        # --- Scaled Dot-Product Attention ---
        attn_scores = torch.sum(center_queries * neighbor_keys, dim=-1) / (self.q_proj.out_features**0.5)
        
        # <<<<<<<<<<<<<<<<< SIMPLIFICATION HERE <<<<<<<<<<<<<<<<<
        # Use the numerically stable scatter_softmax from torch_scatter
        attn_weights = scatter_softmax(attn_scores, center_indices, dim=0)
        
        # --- Aggregate Values using Attention Weights ---
        total_atoms = base_atom_embeds.size(0)
        attn_output = scatter(
            attn_weights.unsqueeze(-1) * neighbor_values, 
            center_indices, 
            dim=0, 
            dim_size=total_atoms, 
            reduce="sum"
        )
        
        # --- Final Refinement ---
        refined_embeds = self.layer_norm(base_atom_embeds + attn_output)
        final_embeds = self.final_mlp(refined_embeds)
        
        return final_embeds