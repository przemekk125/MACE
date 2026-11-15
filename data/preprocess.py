from __future__ import annotations

import torch
from torch_geometric.data import Data


def build_edge_index(
    pos: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a directed radius graph from atomic positions.

    An edge ``(i, j)`` is included when atoms ``i`` and ``j`` belong to the
    same graph, are distinct, and their distance is at most ``cutoff``.
    """
    if pos.ndim != 2:
        raise ValueError(f"pos must have shape [num_nodes, dimensions], got {pos.shape}")
    if not torch.is_floating_point(pos):
        raise TypeError("pos must be a floating-point tensor")
    if not torch.isfinite(torch.tensor(cutoff)):
        raise ValueError("cutoff must be finite")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    num_nodes = pos.shape[0]
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=pos.device)
    elif batch.ndim != 1 or batch.shape[0] != num_nodes:
        raise ValueError("batch must have shape [num_nodes]")

    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)

    displacement = pos[:, None, :] - pos[None, :, :]
    squared_distances = displacement.square().sum(dim=-1)
    connected = (
        (squared_distances <= cutoff**2)
        & (squared_distances > 0)
        & (batch[:, None] == batch[None, :])
    )
    return connected.nonzero(as_tuple=False).t().contiguous()


class NeighborhoodGraph:
    """PyG transform that adds a cutoff-based directed neighbourhood graph."""

    def __init__(self, cutoff: float):
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = cutoff

    def __call__(self, data: Data) -> Data:
        data.edge_index = build_edge_index(data.pos, self.cutoff)
        return data