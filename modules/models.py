from collections.abc import Sequence

import cuequivariance as cue
import torch


class MACE(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: cue.Irreps,
        node_feats_irreps: cue.Irreps,
        edge_attrs_irreps: cue.Irreps,
        edge_feats_irreps: cue.Irreps,
        hidden_irreps: Sequence[cue.Irreps],
        mid_interaction_irreps: Sequence[cue.Irreps],
        interaction_irreps_out: Sequence[cue.Irreps],
        irreps_out: Sequence[cue.Irreps],
        radial_mlp_hidden_dims: Sequence[Sequence[int]] = ((64,), (64,)),
    ):
        super().__init__()

    def forward():
        pass
