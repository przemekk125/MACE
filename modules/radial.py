import numpy as np
import torch


class BesselBasis(torch.nn.Module):
    def __init__(self, r_max: float, num_basis: int = 16):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.arange(1, num_basis + 1) * np.pi / r_max,
        )
        self.weights.requires_grad = False
        self.norm = np.sqrt(2 / r_max)

    def forward(self, r: torch.Tensor):
        r = r[...,None]
        weights = self.weights[None,:]
        return torch.where(
            r > 0,
            self.norm * (torch.sin(weights * r) / r),
            self.norm * self.weights * torch.ones_like(r),
        )


class PolynomialCutoff(torch.nn.Module):
    def __init__(self, r_max: float, p: float):
        super().__init__()
        self.p = p
        self.r_max = r_max

    def forward(self, r: torch.Tensor):
        x = r / self.r_max
        p = self.p
        inside = (
            1
            - (p + 1) * (p + 2) / 2 * x**p
            + p * (p + 2) * x ** (p + 1)
            - p * (p + 1) / 2 * x ** (p + 2)
        )
        return torch.where(x <= 1, inside, torch.zeros_like(x))
