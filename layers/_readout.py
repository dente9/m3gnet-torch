# m3gnet/layers/_readout.py

"""Readout layers for pooling graph information."""

from typing import List
import torch
import torch.nn as nn
from torch_scatter import scatter

from ._core import MLP

def _unsorted_segment_softmax(data, segment_ids, num_segments):
    """Custom softmax for scatter operation."""
    maxes = scatter(data, segment_ids, dim=0, dim_size=num_segments, reduce="max")
    data_exp = torch.exp(data - maxes[segment_ids])
    data_sum = scatter(data_exp, segment_ids, dim=0, dim_size=num_segments, reduce="sum")
    return data_exp / (data_sum[segment_ids] + 1e-16)

class ReduceReadOut(nn.Module):
    """Reduces node or edge features to a graph-level feature vector."""
    def __init__(self, reducer: str = "mean"):
        super().__init__()
        self.reducer = reducer

    def forward(self, features: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        return scatter(features, batch_index, dim=0, reduce=self.reducer)

class WeightedReadout(nn.Module):
    """Weighted readout layer."""
    def __init__(self, neurons: List[int], activation=nn.SiLU()):
        super().__init__()
        self.feature_mlp = MLP(neurons, activation)
        # Weighting MLP ends with a single output for the weight
        weight_neurons = neurons[:-1] + [1]
        self.weight_mlp = MLP(weight_neurons, activation)

    def forward(self, features: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        num_graphs = torch.max(batch_index) + 1
        
        # Get per-node/edge features
        processed_features = self.feature_mlp(features)
        
        # Get per-node/edge weights
        weights = self.weight_mlp(features)
        
        # Apply softmax weighting within each graph
        softmax_weights = _unsorted_segment_softmax(weights, batch_index, num_graphs)
        
        # Weighted sum
        return scatter(processed_features * softmax_weights, batch_index, dim=0, reduce="sum")


class Set2Set(nn.Module):
    """Set2Set readout layer."""
    def __init__(self, in_features: int, processing_steps: int, num_layers: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = 2 * in_features
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.out_features, self.in_features, num_layers)

    def forward(self, features: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        num_graphs = torch.max(batch_index) + 1
        
        h = (torch.zeros(self.num_layers, num_graphs, self.in_features, device=features.device),
             torch.zeros(self.num_layers, num_graphs, self.in_features, device=features.device))
        
        q_star = torch.zeros(num_graphs, self.out_features, device=features.device)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.squeeze(0)
            
            # Attention mechanism
            e = (features * q[batch_index]).sum(dim=-1, keepdim=True)
            a = _unsorted_segment_softmax(e, batch_index, num_graphs)
            
            # Apply attention
            r = scatter(a * features, batch_index, dim=0, reduce="sum")
            q_star = torch.cat([q, r], dim=-1)
            
        return q_star