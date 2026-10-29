"""Segregated PINN: one MLP per physical field, with hard-enforced initial conditions.

Following Laubscher 2021 (PINN-3 architecture), we use separate sub-networks per
physics group. For the Luikov system, the natural split is one network for T and
one for M, since the two fields evolve on different time scales (separated by Lu).
A monolithic single-output network forces gradient balancing across that scale gap;
the segregated form lets each sub-network specialize.

Hard-enforced ICs (`hard_ic=True`, default): the architecture multiplies the raw
network outputs by tau, so T(xi, 0) = 0 and M(xi, 0) = 1 are exactly satisfied to
machine precision. This removes the IC loss from the optimization landscape, which
is the single largest convergence improvement available for smooth-solution PINNs.
The autograd graph still works correctly: dT/dtau, d2T/dxi2, etc. all compose via
the chain rule. See e.g. Lagaris et al. 1998, Berg & Nystrom 2018, Sukumar & Srivastava
2022 for the technique.

Coupling between T and M happens in the loss function (the PDE residual contains both
fields), not in the architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from kibapap.models.mlp import MLP


class SegregatedPINN(nn.Module):
    """Two MLPs, one for each physical field. Returns (T_pred, M_pred).

    With `hard_ic=True` (default), the IC T(xi, 0)=0 and M(xi, 0)=1 are enforced by
    the architecture via output scaling:

        T(xi, tau) = tau * T_net(x)
        M(xi, tau) = 1 + tau * M_net(x)

    The network learns the "delta" rather than the absolute fields, and the IC loss
    becomes identically zero by construction.

    Parameters
    ----------
    in_dim : int
        2 for non-parametric (xi, tau); 6 for parametric (xi, tau, Bi_t, Bi_m, Lu, Ko).
        The hard-IC mechanism reads tau from index 1 regardless.
    hidden_dim : int
        Width of hidden layers. Same for both sub-networks.
    depth : int
        Number of hidden layers in each sub-network.
    fourier : bool
        Whether to apply Fourier feature embedding. Default True.
    fourier_features : int
        Number of Fourier feature pairs.
    fourier : bool
        Whether to apply Fourier feature embedding. For the smooth Luikov baseline,
        leave this False -- the fields carry 97% of their energy in the lowest spatial
        modes, and a high-bandwidth embedding makes the d2/dxi2 PDE residual explode at
        init (~1e6) and stalls training. See FourierFeatures docstring.
    fourier_sigma : float
        Bandwidth of the random projection. Keep <= 0.5 if Fourier is enabled on the
        smooth baseline; only raise it for genuinely sharp-front Phase-2 cases.
    hard_ic : bool
        If True (default), enforce IC architecturally via output scaling.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        depth: int = 5,
        fourier: bool = True,
        fourier_features: int = 32,
        fourier_sigma: float = 0.5,
        hard_ic: bool = True,
    ) -> None:
        super().__init__()
        self.hard_ic = hard_ic
        self.in_dim = in_dim
        # Different seeds so the two sub-networks have different Fourier projections
        self.T_net = MLP(
            in_dim,
            out_dim=1,
            hidden_dim=hidden_dim,
            depth=depth,
            fourier=fourier,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            seed=0,
        )
        self.M_net = MLP(
            in_dim,
            out_dim=1,
            hidden_dim=hidden_dim,
            depth=depth,
            fourier=fourier,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            seed=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (..., in_dim) -> (T: (..., 1), M: (..., 1))."""
        T_raw = self.T_net(x)
        M_raw = self.M_net(x)
        if self.hard_ic:
            tau = x[..., 1:2]
            # T(xi, 0) = 0, M(xi, 0) = 1 by construction.
            T = tau * T_raw
            M = 1.0 + tau * M_raw
            return T, M
        return T_raw, M_raw
