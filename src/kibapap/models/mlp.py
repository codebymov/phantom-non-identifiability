"""Building blocks: random Fourier feature embedding and a configurable MLP.

The Fourier feature embedding addresses the spectral bias of plain MLPs documented in
Tancik et al. 2020 and Wang et al. 2021 for PINNs. Without it, networks struggle to
fit the high-frequency components that appear at sharp evaporation fronts.

The MLP uses tanh activations (standard in PINN literature; sin and SiLU are alternatives
that mainly matter for second-order autodiff smoothness) and Glorot normal init.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Random Gaussian Fourier feature embedding.

    Maps an input vector x of dimension `in_dim` to a 2 * num_features dimensional
    vector [sin(2*pi * B x), cos(2*pi * B x)] where B is a fixed (non-trainable)
    Gaussian random matrix with standard deviation `sigma`. Higher sigma encodes
    higher frequencies, which helps fit sharp gradients but can hurt convergence on
    smooth solutions.

    WARNING: because the PINN loss differentiates the network twice (d2/dxi2 in the
    PDE residual), the residual at init scales like the embedding frequency SQUARED.
    With the 2*pi factor below, sigma=2 reaches effective frequencies ~43, giving an
    init PDE residual ~1e6 that Adam cannot descend -- the network plateaus at a
    smooth-but-wrong solution (see runs/baseline_v2: rel_l2_T blew up to 3.8). The
    Luikov drying fields are smooth (97% of spatial energy in modes 0-3), so Fourier
    features are unnecessary here: prefer fourier=False, or sigma <= 0.5 if enabled.
    Only raise sigma for genuinely sharp-front cases, and watch the init residual.

    Parameters
    ----------
    in_dim : int
        Number of input coordinates (e.g. 2 for (xi, tau), 6 if parametric Luikov).
    num_features : int
        Number of Fourier feature pairs. Output dimension is 2 * num_features.
    sigma : float
        Standard deviation of the Gaussian random projection matrix.
    seed : int | None
        Seed for the random projection. Fixing this is important for reproducibility
        because the projection is non-trainable.
    """

    def __init__(self, in_dim: int, num_features: int = 32, sigma: float = 0.5, seed: int | None = 0) -> None:
        super().__init__()
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            B = sigma * torch.randn(num_features, in_dim, generator=g)
        else:
            B = sigma * torch.randn(num_features, in_dim)
        self.register_buffer("B", B)
        self.in_dim = in_dim
        self.out_dim = 2 * num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) -> (..., 2 * num_features)."""
        proj = 2.0 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    """Configurable MLP with optional Fourier embedding and Glorot init.

    Parameters
    ----------
    in_dim : int
        Input dimension. For Phase-1 non-parametric, in_dim = 2 (xi, tau).
        For parametric (Manavi-style), in_dim = 6 (xi, tau, Bi_t, Bi_m, Lu, Ko).
    out_dim : int
        Output dimension. 1 for a per-field network in the segregated architecture.
    hidden_dim : int
        Width of hidden layers. 64 is the phase-1 default; raise to 128-256 if loss
        plateaus prematurely.
    depth : int
        Number of hidden layers (the network has depth + 1 weight matrices total).
        4-6 is the typical PINN range.
    fourier : bool
        Whether to apply random Fourier feature embedding to the input.
    fourier_features : int
        Number of Fourier feature pairs (output is 2x this).
    fourier_sigma : float
        Bandwidth of the random projection.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int = 64,
        depth: int = 5,
        fourier: bool = True,
        fourier_features: int = 32,
        fourier_sigma: float = 0.5,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if fourier:
            self.embed: nn.Module = FourierFeatures(in_dim, fourier_features, fourier_sigma, seed=seed)
            actual_in = self.embed.out_dim
        else:
            self.embed = nn.Identity()
            actual_in = in_dim

        layers: list[nn.Module] = []
        prev = actual_in
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.Tanh())
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

        # Glorot normal init for tanh nets (gain=5/3 per PyTorch convention)
        gain = nn.init.calculate_gain("tanh")
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(x))
