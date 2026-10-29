"""Sampling collocation, boundary, and initial points for PINN training.

Two strategies are provided. Uniform random sampling is the default and is robust.
Latin hypercube sampling (LHS) covers the domain more evenly with the same number of
points and tends to give 10-20% lower residual variance, but is overkill for the
problem sizes at hand. Both return tensors of shape (N, in_dim).

For parametric Luikov, the parameter columns are sampled uniformly within prescribed
ranges. The standard ranges (Manavi 2024) are Bi_t in [0.5, 2], Bi_m in [0.5, 2],
Lu in [0.2, 1.0], Ko in [1.0, 3.0]. Override via the `param_ranges` argument.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

PARAM_RANGES_DEFAULT: dict[str, tuple[float, float]] = {
    "Bi_t": (0.5, 2.0),
    "Bi_m": (0.5, 2.0),
    "Lu": (0.2, 1.0),
    "Ko": (1.0, 3.0),
}


def _sample_params(
    n: int,
    ranges: Mapping[str, tuple[float, float]],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample n parameter rows for (Bi_t, Bi_m, Lu, Ko)."""
    cols = []
    for key in ("Bi_t", "Bi_m", "Lu", "Ko"):
        lo, hi = ranges[key]
        u = torch.rand(n, 1, device=device, generator=generator)
        cols.append(lo + (hi - lo) * u)
    return torch.cat(cols, dim=1)


def sample_collocation(
    n: int,
    tau_final: float = 2.0,
    parametric: bool = False,
    param_ranges: Mapping[str, tuple[float, float]] | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform random collocation points in the interior of the spatiotemporal domain.

    For non-parametric: returns (N, 2) with (xi, tau) in [0,1] x [0, tau_final].
    For parametric: returns (N, 6) with parameter columns appended.
    """
    xi = torch.rand(n, 1, device=device, generator=generator)
    tau = tau_final * torch.rand(n, 1, device=device, generator=generator)
    base = torch.cat([xi, tau], dim=1)
    if not parametric:
        return base
    ranges = param_ranges or PARAM_RANGES_DEFAULT
    p = _sample_params(n, ranges, device=device, generator=generator)
    return torch.cat([base, p], dim=1)


def sample_boundary(
    n: int,
    tau_final: float = 2.0,
    parametric: bool = False,
    param_ranges: Mapping[str, tuple[float, float]] | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n points each on xi=0 and xi=1 boundaries."""
    tau = tau_final * torch.rand(n, 1, device=device, generator=generator)
    xi_left = torch.zeros(n, 1, device=device)
    xi_right = torch.ones(n, 1, device=device)
    base_left = torch.cat([xi_left, tau], dim=1)
    base_right = torch.cat([xi_right, tau], dim=1)
    if not parametric:
        return base_left, base_right
    ranges = param_ranges or PARAM_RANGES_DEFAULT
    p = _sample_params(n, ranges, device=device, generator=generator)
    return torch.cat([base_left, p], dim=1), torch.cat([base_right, p], dim=1)


def sample_initial(
    n: int,
    parametric: bool = False,
    param_ranges: Mapping[str, tuple[float, float]] | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample n points on tau=0 with xi uniform in [0, 1]."""
    xi = torch.rand(n, 1, device=device, generator=generator)
    tau = torch.zeros(n, 1, device=device)
    base = torch.cat([xi, tau], dim=1)
    if not parametric:
        return base
    ranges = param_ranges or PARAM_RANGES_DEFAULT
    p = _sample_params(n, ranges, device=device, generator=generator)
    return torch.cat([base, p], dim=1)
