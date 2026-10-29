"""PINN loss components: PDE residuals, BC residuals, IC residuals, gPINN extension.

These functions assume the dimensionless Luikov system defined in `kibapap.physics.luikov`:

    dT/dtau  = d2T/dxi2 + Ko * dM/dtau
    dM/dtau  = Lu * d2M/dxi2

with Robin BC at xi=0 ( dT/dxi = Bi_t (T-1), dM/dxi = Bi_m M ),
Neumann at xi=1 (zero gradient), and IC T(xi,0)=0, M(xi,0)=1.

All gradients use torch.autograd to keep the implementation parameter-agnostic; this
extends to phase 4 (web motion) without changes to the loss code, only to the residual
expressions.

Convention: input tensors x have shape (N, in_dim) where the first two columns are
always (xi, tau). For parametric versions (in_dim > 2) the remaining columns are
(Bi_t, Bi_m, Lu, Ko).
"""

from __future__ import annotations

import torch
from torch import nn

from kibapap.physics.luikov import LuikovParams


def _grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute dy/dx for a scalar-summed y. Returns same shape as x."""
    return torch.autograd.grad(
        outputs=y.sum(),
        inputs=x,
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def _extract_params(
    x: torch.Tensor, defaults: LuikovParams
) -> tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]:
    """Pull (Bi_t, Bi_m, Lu, Ko) from the input columns if parametric, else use defaults.

    For non-parametric Phase-1 use, defaults are baseline scalars. For parametric Phase-1b,
    the network input contains the four parameters as additional columns and we extract
    them per-sample.
    """
    if x.shape[-1] >= 6:
        # parametric ordering: (xi, tau, Bi_t, Bi_m, Lu, Ko)
        Bi_t = x[..., 2:3]
        Bi_m = x[..., 3:4]
        Lu = x[..., 4:5]
        Ko = x[..., 5:6]
    else:
        Bi_t = defaults.Bi_t
        Bi_m = defaults.Bi_m
        Lu = defaults.Lu
        Ko = defaults.Ko
    return Bi_t, Bi_m, Lu, Ko


def pde_residuals(
    model: nn.Module,
    x: torch.Tensor,
    defaults: LuikovParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (residual_T, residual_M) at collocation points x.

    Each residual has shape (N, 1) and should be driven to zero by training.

    If x already requires grad (e.g. gPINN differentiates the residual w.r.t. the
    same x), it is used directly so the residual stays connected to that tensor;
    otherwise a leaf copy that requires grad is made.
    """
    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)
    T, M = model(x)
    grads_T = _grad(T, x)
    grads_M = _grad(M, x)
    T_xi = grads_T[..., 0:1]
    T_tau = grads_T[..., 1:2]
    M_xi = grads_M[..., 0:1]
    M_tau = grads_M[..., 1:2]

    # Second spatial derivatives via second autograd pass
    T_xixi = _grad(T_xi, x)[..., 0:1]
    M_xixi = _grad(M_xi, x)[..., 0:1]

    _, _, Lu, Ko = _extract_params(x, defaults)

    res_T = T_tau - T_xixi - Ko * M_tau
    res_M = M_tau - Lu * M_xixi
    return res_T, res_M


def boundary_residuals(
    model: nn.Module,
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    defaults: LuikovParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the four BC residuals.

    Returns
    -------
    res_T_left, res_M_left : (N, 1)
        Robin residuals at xi=0:
        dT/dxi - Bi_t * (T - 1) and dM/dxi - Bi_m * M.
    res_T_right, res_M_right : (N, 1)
        Neumann residuals at xi=1: dT/dxi and dM/dxi.
    """
    # Surface (xi=0): Robin
    x_left = x_left.detach().clone().requires_grad_(True)
    T_l, M_l = model(x_left)
    T_xi_l = _grad(T_l, x_left)[..., 0:1]
    M_xi_l = _grad(M_l, x_left)[..., 0:1]
    Bi_t, Bi_m, _, _ = _extract_params(x_left, defaults)
    res_T_left = T_xi_l - Bi_t * (T_l - 1.0)
    res_M_left = M_xi_l - Bi_m * M_l

    # Back (xi=1): Neumann (zero gradient)
    x_right = x_right.detach().clone().requires_grad_(True)
    T_r, M_r = model(x_right)
    T_xi_r = _grad(T_r, x_right)[..., 0:1]
    M_xi_r = _grad(M_r, x_right)[..., 0:1]
    res_T_right = T_xi_r
    res_M_right = M_xi_r

    return res_T_left, res_M_left, res_T_right, res_M_right


def initial_residuals(model: nn.Module, x_ic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial condition residuals: T(xi, 0) = 0, M(xi, 0) = 1.

    x_ic should have tau = 0 for all rows. This function does not enforce that,
    so the caller is responsible for sampling correctly.
    """
    T, M = model(x_ic)
    return T - 0.0, M - 1.0


def measurement_consistency_residuals(
    model: nn.Module,
    meas: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measurement-consistency (Sobolev-data) residuals -- the differentiator from gPINN.

    gPINN (`gpinn_residual_gradient`) penalizes gradients of the *PDE residual*: a
    physics-side regularizer that needs no data. Here we instead supervise the
    network's *output* integral and time-derivative with quantities a real drying
    cell measures:

      - integral observation  : the weighing chamber reads the through-thickness mean
        moisture  M_bar(t) = <M(.,t)>_xi  (gravimetric).
      - derivative observation : mass-flow sensors read the surface drying rate
        dM/dt|_(xi=0)(t).

    These are integral/derivative observations of the field, not the field itself, so
    the term is orthogonal to both a plain data-fit and gPINN's residual-gradient term.
    Setting its weight to 0 recovers the model without it (the ablation baseline).

    Parameters
    ----------
    meas : dict with tensors
        t_meas      : (K, 1) measurement times.
        M_integral  : (K, 1) measured through-thickness mean moisture at t_meas.
        dMdt_surface: (K, 1) measured surface drying rate at t_meas.
        xi_quad     : (Q, 1) quadrature nodes in xi for the mean (e.g. linspace(0,1,Q)).

    Returns
    -------
    res_integral : (K, 1)  predicted mean moisture minus measured.
    res_derivative : (K, 1)  predicted surface dM/dt minus measured.
    """
    t_meas = meas["t_meas"]
    xi_quad = meas["xi_quad"]
    K = t_meas.shape[0]
    Q = xi_quad.shape[0]

    # --- integral observation: <M(.,t)>_xi at each measurement time ---
    # Build the (K*Q, 2) grid of (xi_quad, t_meas) without tracking grad in xi.
    xi_rep = xi_quad.reshape(1, Q, 1).expand(K, Q, 1).reshape(K * Q, 1)
    t_rep = t_meas.reshape(K, 1, 1).expand(K, Q, 1).reshape(K * Q, 1)
    _, M_q = model(torch.cat([xi_rep, t_rep], dim=-1))
    M_bar = M_q.reshape(K, Q).mean(dim=1, keepdim=True)  # (K, 1)
    res_integral = M_bar - meas["M_integral"]

    # --- derivative observation: dM/dt at the surface xi=0 ---
    t_s = t_meas.detach().clone().requires_grad_(True)
    xi_s = torch.zeros_like(t_s)
    _, M_s = model(torch.cat([xi_s, t_s], dim=-1))
    dMdt_s = _grad(M_s, t_s)[..., 0:1]
    res_derivative = dMdt_s - meas["dMdt_surface"]

    return res_integral, res_derivative


def gpinn_residual_gradient(
    model: nn.Module,
    x: torch.Tensor,
    defaults: LuikovParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gPINN extension (Yu et al. 2022).

    If the PDE residual r(x) is exactly zero on the manifold, then nabla r(x) is also
    zero. Penalizing nabla r in the loss enforces a stronger consistency on a finite
    discretization. This is a phase-2 deliverable; included here so the architecture
    is in place from the start.

    Returns
    -------
    grad_res_T, grad_res_M : (N, 1)
        L2 norms of the spatial gradient of the PDE residuals at each collocation point.
    """
    x = x.detach().clone().requires_grad_(True)
    res_T, res_M = pde_residuals(model, x, defaults)
    # Spatial gradient of the residual (we only penalize spatial because temporal
    # smoothness is already implicit in the autograd time derivative.)
    grad_res_T = _grad(res_T, x)[..., 0:1]
    grad_res_M = _grad(res_M, x)[..., 0:1]
    return grad_res_T, grad_res_M
