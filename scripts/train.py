import argparse
import random
from pathlib import Path

import cuequivariance as cue
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from modules.models import MACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MACE on QM9.")
    parser.add_argument("--data-root", type=Path, default=Path("data/qm9"))
    parser.add_argument("--checkpoint", type=Path, default=Path("trained/mace_qm9.pt"))
    parser.add_argument("--target-index", type=int, default=7)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--cutoff-p", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def build_model(r_max: float, cutoff_p: float) -> MACE:
    num_elements = 5
    state_irreps = cue.Irreps("O3", "8x0e + 8x1o + 8x2e")
    edge_attrs_irreps = cue.Irreps("O3", "1x0e + 1x1o + 1x2e + 1x3o")
    edge_feats_irreps = cue.Irreps("O3", "16x0e")

    return MACE(
        node_attrs_irreps=cue.Irreps("O3", "5x0e"),
        node_feats_irreps=[state_irreps, state_irreps],
        edge_attrs_irreps=edge_attrs_irreps,
        edge_feats_irreps=edge_feats_irreps,
        hidden_irreps=[state_irreps, state_irreps],
        mid_interaction_irreps=[state_irreps, state_irreps],
        interaction_irreps_out=[state_irreps, state_irreps],
        irreps_out=[state_irreps, state_irreps],
        contraction_degrees=[2, 3],
        num_elements=num_elements,
        atomic_numbers=[1, 6, 7, 8, 9],
        r_max=r_max,
        cutoff_p=cutoff_p,
        output_dim=1,
    )


def run_epoch(
    model: MACE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    target_index: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_normalized_loss = 0.0
    total_physical_loss = 0.0
    total_graphs = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch).squeeze(-1)
            target = batch.y[:, target_index]
            normalized_target = (target - target_mean) / target_std
            loss = torch.nn.functional.mse_loss(prediction, normalized_target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            physical_prediction = prediction * target_std + target_mean
            physical_loss = torch.nn.functional.mse_loss(
                physical_prediction, target
            )
            graph_count = batch.num_graphs
            total_normalized_loss += loss.item() * graph_count
            total_physical_loss += physical_loss.item() * graph_count
            total_graphs += graph_count

    return (
        total_normalized_loss / total_graphs,
        total_physical_loss / total_graphs,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset = QM9(
        root=str(args.data_root))
    if not 0 <= args.target_index < dataset.num_classes:
        raise ValueError(
            f"target-index must be in [0, {dataset.num_classes - 1}]"
        )

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(dataset), generator=generator)
    train_end = min(args.train_size, len(dataset))
    validation_end = min(train_end + args.validation_size, len(dataset))
    train_dataset = dataset[permutation[:train_end]]
    validation_dataset = dataset[permutation[train_end:validation_end]]
    train_targets = torch.cat(
        [sample.y[:, args.target_index] for sample in train_dataset]
    )
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std().clamp_min(
        torch.finfo(train_targets.dtype).eps
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.r_max, args.cutoff_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_normalized_mse, train_physical_mse = run_epoch(
            model,
            train_loader,
            optimizer,
            args.target_index,
            target_mean,
            target_std,
            device,
        )
        validation_normalized_mse, validation_physical_mse = run_epoch(
            model,
            validation_loader,
            None,
            args.target_index,
            target_mean,
            target_std,
            device,
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train normalized MSE {train_normalized_mse:.6e} | "
            f"validation normalized MSE {validation_normalized_mse:.6e} | "
            f"train physical MSE {train_physical_mse:.6e} | "
            f"validation physical MSE {validation_physical_mse:.6e}"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "target_mean": target_mean.cpu(),
            "target_std": target_std.cpu(),
        },
        args.checkpoint,
    )


if __name__ == "__main__":
    main()