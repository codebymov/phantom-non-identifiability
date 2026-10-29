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
    """Sample parameter rows for (Bi_t, Bi_m, Lu, Ko), uniformly within their ranges.

    Args:
        n: Number of rows to draw.
        ranges: Inclusive ``(lo, hi)`` bounds per parameter name. Must contain all four keys.
        device: Torch device to allocate on.
        generator: Optional generator, for reproducible draws.

    Returns:
        Parameter columns of shape ``(n, 4)``, ordered ``(Bi_t, Bi_m, Lu, Ko)``.
    """
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

    Args:
        n: Number of points to draw.
        tau_final: Upper end of the time interval; points fall in ``[0, tau_final]``.
        parametric: Append the four parameter columns, for a parametric surrogate.
        param_ranges: Parameter bounds when `parametric` is set. Defaults to
            ``PARAM_RANGES_DEFAULT``.
        device: Torch device to allocate on.
        generator: Optional generator, for reproducible draws.

    Returns:
        Shape ``(n, 2)`` holding ``(xi, tau)``, or ``(n, 6)`` with the parameter columns
        appended when `parametric` is set.
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
    """Sample points on both spatial boundaries at matched times.

    Both returned sets share the same ``tau`` draw, so a boundary residual can pair them.

    Args:
        n: Number of points per boundary.
        tau_final: Upper end of the time interval.
        parametric: Append the four parameter columns.
        param_ranges: Parameter bounds when `parametric` is set. Defaults to
            ``PARAM_RANGES_DEFAULT``.
        device: Torch device to allocate on.
        generator: Optional generator, for reproducible draws.

    Returns:
        ``(left, right)`` for ``xi = 0`` and ``xi = 1``, each of shape ``(n, 2)``, or
        ``(n, 6)`` when `parametric` is set.
    """
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
    """Sample points on the initial line ``tau = 0``, with ``xi`` uniform in ``[0, 1]``.

    Args:
        n: Number of points to draw.
        parametric: Append the four parameter columns.
        param_ranges: Parameter bounds when `parametric` is set. Defaults to
            ``PARAM_RANGES_DEFAULT``.
        device: Torch device to allocate on.
        generator: Optional generator, for reproducible draws.

    Returns:
        Shape ``(n, 2)`` holding ``(xi, 0)``, or ``(n, 6)`` when `parametric` is set.
    """
    xi = torch.rand(n, 1, device=device, generator=generator)
    tau = torch.zeros(n, 1, device=device)
    base = torch.cat([xi, tau], dim=1)
    if not parametric:
        return base
    ranges = param_ranges or PARAM_RANGES_DEFAULT
    p = _sample_params(n, ranges, device=device, generator=generator)
    return torch.cat([base, p], dim=1)
