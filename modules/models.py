from collections.abc import Sequence

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn.functional as F
from cuequivariance_torch import SphericalHarmonics

from .blocks import InteractionBlock, SymmetricContractionBlock, MaceHead
from .radial import BesselBasis, PolynomialCutoff
from .utils import get_vectors_and_lengths


class MACE(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: cue.Irreps,
        node_feats_irreps: Sequence[cue.Irreps],
        edge_attrs_irreps: cue.Irreps,
        edge_feats_irreps: cue.Irreps,
        hidden_irreps: Sequence[cue.Irreps],
        mid_interaction_irreps: Sequence[cue.Irreps],
        interaction_irreps_out: Sequence[cue.Irreps],
        irreps_out: Sequence[cue.Irreps],
        contraction_degrees: Sequence[int],
        num_elements: int,
        atomic_numbers: Sequence[int],
        r_max: float,
        cutoff_p: float,
        output_dim: int = 1,
        num_radial_basis: int = 16,
        spherical_harmonics_lmax: int = 3,
        radial_mlp_hidden_dims: Sequence[Sequence[int]] = ((64,), (64,)),
    ):
        super().__init__()
        if len(atomic_numbers) != num_elements:
            raise ValueError("atomic_numbers must contain one entry per element")
        if len(set(atomic_numbers)) != num_elements:
            raise ValueError("atomic_numbers must contain unique values")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.num_elements = num_elements
        self.register_buffer(
            "atomic_numbers",
            torch.tensor(atomic_numbers, dtype=torch.long),
        )
        num_interactions = len(hidden_irreps)
        if len(node_feats_irreps) != num_interactions:
            raise ValueError(
                "node_feats_irreps must contain one output state irrep per "
                "interaction"
            )
        initial_node_feats_irreps = cue.Irreps("O3", f"{num_elements}x0e")
        node_state_irreps = (initial_node_feats_irreps, *node_feats_irreps)
        parameter_sequences = (
            mid_interaction_irreps,
            interaction_irreps_out,
            irreps_out,
            contraction_degrees,
            radial_mlp_hidden_dims,
        )
        if any(len(values) != num_interactions for values in parameter_sequences):
            raise ValueError(
                "hidden_irreps, mid_interaction_irreps, interaction_irreps_out, "
                "irreps_out, contraction_degrees, and radial_mlp_hidden_dims "
                "must have the same length"
            )
        # this is scalar irrep:
        if edge_feats_irreps.dim != num_radial_basis:
            raise ValueError(
                "edge_feats_irreps must contain one scalar channel per radial basis "
                "function"
            )

        self.length_encoder = BesselBasis(r_max, num_radial_basis)
        self.cutoff = PolynomialCutoff(r_max, cutoff_p)
        self.harmonics = SphericalHarmonics(list(range(spherical_harmonics_lmax + 1)))

        self.interaction_blocks = torch.nn.ModuleList(
            InteractionBlock(
                node_feats_irreps=node_state_irreps[index],
                edge_attrs_irreps=edge_attrs_irreps,
                edge_feats_irreps=edge_feats_irreps,
                hidden_irreps=hidden_irreps[index],
                mid_irreps=mid_interaction_irreps[index],
                target_irreps=interaction_irreps_out[index],
                radial_mlp_hidden_dims=radial_mlp_hidden_dims[index],
            )
            for index in range(num_interactions)
        )
        self.message_linear_layers = torch.nn.ModuleList(
            cuet.Linear(irreps_out[index], node_state_irreps[index + 1])
            for index in range(num_interactions)
        )
        self.node_linear_layers = torch.nn.ModuleList(
            cuet.Linear(node_state_irreps[index], node_state_irreps[index + 1])
            for index in range(num_interactions)
        )
        self.head = MaceHead(node_state_irreps, output_dim)
        self.contraction_blocks = torch.nn.ModuleList(
            SymmetricContractionBlock(
                irreps_in=interaction_irreps_out[index],
                irreps_out=irreps_out[index],
                contraction_degree=contraction_degrees[index],
                num_elements=num_elements,
            )
            for index in range(num_interactions)
        )

    def forward(self, batch):
        vectors, lengths = get_vectors_and_lengths(batch.pos, batch.edge_index)
        radial_features = self.length_encoder(lengths)
        cutoff_values = self.cutoff(lengths).unsqueeze(-1)
        edge_feats = radial_features * cutoff_values
        edge_attrs = self.harmonics(vectors)
        matches = batch.z[:, None] == self.atomic_numbers[None, :]
        if not torch.all(matches.any(dim=1)):
            unknown = torch.unique(batch.z[~matches.any(dim=1)]).tolist()
            raise ValueError(f"Unknown atomic numbers in batch: {unknown}")
        node_attrs = matches.to(torch.long).argmax(dim=1).to(torch.int32)
        node_feats = F.one_hot(
            node_attrs.to(dtype=torch.long),
            num_classes=self.num_elements,
        ).to(dtype=batch.pos.dtype)
        node_states = [node_feats]

        for index, (interaction, contraction) in enumerate(
            zip(self.interaction_blocks, self.contraction_blocks)
        ):
            message = interaction(node_feats, edge_attrs, edge_feats, batch.edge_index)
            message = contraction(message, node_attrs)
            node_feats = self.node_linear_layers[index](node_feats) + self.message_linear_layers[index](message)
            node_states.append(node_feats)

        return self.head(node_states, batch.batch, batch.num_graphs)
