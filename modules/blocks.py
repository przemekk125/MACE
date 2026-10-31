from collections.abc import Sequence

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch

from .utils import neighbour_sum


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: cue.Irreps,
        node_feats_irreps: cue.Irreps,
        edge_attrs_irreps: cue.Irreps,
        edge_feats_irreps: cue.Irreps,
        hidden_irreps: cue.Irreps,
        mid_irreps: cue.Irreps,
        target_irreps: cue.Irreps,
        radial_mlp_hidden_dims: Sequence[int] = (64,),
    ):
        super().__init__()
        # elements encoded
        self.node_attrs_irreps = node_attrs_irreps
        # hidden states
        self.node_feats_irreps = node_feats_irreps
        # spherical harmonics
        self.edge_attrs_irreps = edge_attrs_irreps
        # radial basis encoding
        self.edge_feats_irreps = edge_feats_irreps
        # irreps for hidden state
        self.hidden_irreps = hidden_irreps
        # irreps after the tensor product
        self.mid_irreps = mid_irreps
        # target irrep
        self.target_irreps = target_irreps

        self.linear_h = cuet.Linear(self.node_feats_irreps, hidden_irreps)

        self.tp = cuet.ChannelWiseTensorProduct(
            self.hidden_irreps,
            self.edge_attrs_irreps,
            self.mid_irreps,
            shared_weights=False,
        )
        self.mid_irreps = self.tp.irreps_out

        radial_mlp_layers: list[torch.nn.Module] = []
        input_dim = self.edge_feats_irreps.num_irreps
        for hidden_dim in radial_mlp_hidden_dims:
            radial_mlp_layers.extend(
                [torch.nn.Linear(input_dim, hidden_dim), torch.nn.SiLU()]
            )
            input_dim = hidden_dim
        radial_mlp_layers.append(torch.nn.Linear(input_dim, self.tp.weight_numel))
        self.radial_mlp = torch.nn.Sequential(*radial_mlp_layers)

        self.linear_out = cuet.Linear(self.mid_irreps, self.target_irreps)

    def forward(
            self,
            #node_attrs: torch.Tensor,
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            edge_feats: torch.Tensor,
            edge_index: torch.Tensor):

        node_feats = self.linear_h(node_feats)
        tp_weights = self.radial_mlp(edge_feats)
        # TODO: make cutoff
        message = self.tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = neighbour_sum(message, edge_index, node_feats.shape[0])
        message = self.linear_out(message)
        return message
