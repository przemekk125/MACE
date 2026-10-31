import cuequivariance as cue
import torch


class InteractionBlock(torch.nn.Module):
    def __innit__(
        self,
        node_attrs_irreps: cue.Irreps,
        node_feats_irreps: cue.Irreps,
        edge_attrs_irreps: cue.Irreps,
        edge_feats_irreps: cue.Irreps,
        target_irreps: cue.Irreps,
        hidden_irreps: cue.Irreps,
    ):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
