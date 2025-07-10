# m3gnet_torch/utils/torch_utils.py
import torch
from torch_scatter import segment_sum_coo
from typing import Tuple 


def get_segment_indices_from_n(n: torch.Tensor) -> torch.Tensor:
    """Creates segment IDs from a tensor of counts."""
    return torch.repeat_interleave(
        torch.arange(len(n), device=n.device),
        repeats=n.cpu()
    )

def get_length(vectors: torch.Tensor) -> torch.Tensor:
    """Calculates the norm of a batch of vectors."""
    return torch.linalg.norm(vectors, dim=-1)

def unsorted_segment_softmax_coo(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """Computes a stable softmax over segments."""
    num_segments = int(segment_ids.max().item()) + 1 if segment_ids.numel() > 0 else 0 
    max_vals = torch.full((num_segments,), -torch.inf, device=data.device, dtype=data.dtype)
    try: # Fallback if scatter_max is not available
        from torch_scatter import scatter_max
        if data.numel() > 0: 
            segment_max_vals, _ = scatter_max(data, segment_ids, dim=0, dim_size=num_segments)
        else:
            segment_max_vals = max_vals 
    except ImportError:
        segment_max_vals = max_vals 
        if data.numel() > 0: 
            for i in range(num_segments):
                mask = (segment_ids == i)
                if mask.any():
                    segment_max_vals[i] = data[mask].max()
    
    if data.numel() == 0: 
        return torch.empty_like(data)

    gathered_max_vals = segment_max_vals[segment_ids]
    data_stabilized = data - gathered_max_vals
    exp_data = torch.exp(data_stabilized)
    exp_sum = segment_sum_coo(exp_data, segment_ids, dim=0, dim_size=num_segments)
    gathered_exp_sum = exp_sum[segment_ids]
    return exp_data / (gathered_exp_sum + 1e-16)

def get_pair_vector_from_graph(graph) -> torch.Tensor:
    """
    Given a graph return pair vectors that form the bonds.
    This is a PyTorch translation of the original TensorFlow function.

    Args:
        graph (MaterialGraph): The material graph.

    Returns:
        torch.Tensor: Pair vectors, shape [n_bonds, 3].
    """
    atom_positions = graph.atom_positions
    bond_atom_indices = graph.bond_atom_indices
    
    if bond_atom_indices is None or bond_atom_indices.numel() == 0:
        return torch.empty(0, 3, device=atom_positions.device, dtype=atom_positions.dtype)
        
    pos1 = atom_positions[bond_atom_indices[:, 0]]
    pos2 = atom_positions[bond_atom_indices[:, 1]]
    
    if graph.lattices is not None and graph.pbc_offsets is not None:
        segment_ids = get_segment_indices_from_n(graph.n_bonds)
        bond_lattices = graph.lattices[segment_ids]
        offsets = torch.matmul(graph.pbc_offsets.float(), bond_lattices)
        pos2 = pos2 + offsets

    diff = pos2 - pos1
    
    # --- 关键修正：确保返回的张量是独立的且形状正确 ---
    # .clone() 创建数据副本，.detach() 将其从计算图中分离，.view(-1, 3) 强制形状
    # 这样，这个 pair_vectors 就不受任何外部污染影响，也不影响外部
    return diff.clone().detach().view(-1, 3) 

def compute_threebody_angles(graph, pair_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a graph and its pair vectors, compute the angles for three-body interactions.
    
    Args:
        graph (MaterialGraph): The material graph.
        pair_vectors (torch.Tensor): Pre-computed pair vectors for all bonds, shape [N_bonds, 3].
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - cos_jik (torch.Tensor): The cosine of the angles for each triplet, shape [N_triples].
            - dummy_phi (torch.Tensor): A tensor of zeros for the phi angle, shape [N_triples].
    """
    if graph.triple_bond_indices is None or graph.triple_bond_indices.numel() == 0:
        return torch.tensor([], device=pair_vectors.device, dtype=torch.float32), \
               torch.tensor([], device=pair_vectors.device, dtype=torch.float32)

    # 确保 pair_vectors 是形状为 [N_bonds, 3] 的张量
    # 这一步在 get_pair_vector_from_graph 已经被强制处理，这里只是再次确认
    if pair_vectors.ndim != 2 or pair_vectors.shape[1] != 3:
        raise ValueError(f"Expected pair_vectors to be [N_bonds, 3], but got {pair_vectors.shape}")

    vij = pair_vectors[graph.triple_bond_indices[:, 0]]
    vik = pair_vectors[graph.triple_bond_indices[:, 1]]
    
    rij = get_length(vij)
    rik = get_length(vik)
    
    cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik + 1e-8) 
    cos_jik = torch.clamp(cos_jik, -1.0, 1.0)
    
    dummy_phi = torch.zeros_like(cos_jik)
    
    return cos_jik, dummy_phi