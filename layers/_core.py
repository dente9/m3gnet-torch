# m3gnet/layers/_core.py (Final Corrected Version)

"""Core neural network building blocks."""

from typing import List, Optional, Callable

import torch
import torch.nn as nn

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
        # Use LazyLinear for the first layer
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
        # is_output=True for gating_mlp ensures no final activation before sigmoid
        self.linear_mlp = MLP(neurons, activation)
        self.gating_mlp = MLP(neurons, activation, is_output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_mlp(x) * torch.sigmoid(self.gating_mlp(x))

class AtomEmbedding(nn.Module):
    def __init__(self, n_atom_types: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, embedding_dim)

    def forward(self, atom_numbers: torch.Tensor) -> torch.Tensor:
        return self.embedding(atom_numbers.long().squeeze(-1))