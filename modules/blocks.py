from collections.abc import Sequence

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch

from .utils import neighbour_sum


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: cue.Irreps,
        edge_attrs_irreps: cue.Irreps,
        edge_feats_irreps: cue.Irreps,
        hidden_irreps: cue.Irreps,
        mid_irreps: cue.Irreps,
        target_irreps: cue.Irreps,
        radial_mlp_hidden_dims: Sequence[int] = (64,),
    ):
        super().__init__()
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
        # node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ):

        node_feats = self.linear_h(node_feats)
        tp_weights = self.radial_mlp(edge_feats)
        # TODO: make cutoff
        message = self.tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = neighbour_sum(message, edge_index, node_feats.shape[0])
        message = self.linear_out(message)
        return message


class SymmetricContractionBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        contraction_degree: int,
        num_elements: int,
    ):
        super().__init__()
        self.contraction = cuet.SymmetricContraction(
            irreps_in, irreps_out, contraction_degree, num_elements
        )

    def forward(self, message, indices):
        return self.contraction(message, indices)

class MaceHead(torch.nn.Module):
    def __init__(
        self,
        node_state_irreps: Sequence[cue.Irreps],
        output_dim: int,
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        output_irreps = cue.Irreps("O3", f"{output_dim}x0e")
        self.output_linear_layers = torch.nn.ModuleList(
            cuet.Linear(state_irreps, output_irreps)
            for state_irreps in node_state_irreps[:-1]
        )
        self.output_mlp = torch.nn.Sequential(
            cuet.Linear(node_state_irreps[-1], output_irreps),
            torch.nn.SiLU(),
            torch.nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        node_states: Sequence[torch.Tensor],
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        graph_outputs = []
        for index, node_state in enumerate(node_states[:-1]):
            output_features = self.output_linear_layers[index](node_state)

            graph_output = torch.zeros(
                num_graphs,
                output_features.shape[-1],
                device=output_features.device,
                dtype=output_features.dtype,
            )
            graph_outputs.append(
                graph_output.index_add_(0, batch_index, output_features)
            )

        output_features = self.output_mlp(node_states[-1])
        graph_output = torch.zeros(
            num_graphs,
            output_features.shape[-1],
            device=output_features.device,
            dtype=output_features.dtype,
        )
        graph_outputs.append(graph_output.index_add_(0, batch_index, output_features))

        return torch.stack(graph_outputs, dim=0).sum(dim=0)