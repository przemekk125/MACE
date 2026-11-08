import torch
from torch_geometric.utils import scatter


def get_vectors_and_lengths(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    vecs = pos[edge_index[0]]-pos[edge_index[1]]
    lengths = torch.linalg.norm(vecs, dim=-1)
    vecs = vecs/lengths.unsqueeze(-1)
    return vecs, lengths


def neighbour_sum(message, edge_index, num_nodes = None):
    receivers = edge_index[1]
    return scatter(src=message,index=receivers,reduce="sum", dim_size=num_nodes)