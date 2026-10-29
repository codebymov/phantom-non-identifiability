"""Differentiable torch mirror of `kibapap.physics.fd_solver.solve_luikov_fd`.

Why a mirror rather than a port: the reference solver uses `scipy.linalg.solve_banded` (LAPACK), and
the existing regression asserting that the `Pn=0` path is *bit-identical* to the original reduced
solver is load-bearing for the predecessor paper. Any reimplementation reorders floating-point
operations, so it cannot be bit-identical. We therefore leave the reference untouched and add this
mirror for tangents only, gated by a standing test that the two agree to ~1e-13 relative
(`tangentsafe/tests/test_mirror_agreement.py`).

Structure follows fd_solver.py line for line:
  implicit Euler, operator-split, two tridiagonal solves per step, with the thermogradient term
  lagged-explicit at the current T (an O(dtau) split, matching the reference).

The spatial operators are time-invariant, so each is LU-factorised once and reused across all time
steps; this is what makes AD through a 200-step rollout cheap.

Differentiable in (Lu, Ko, Bi_t, Bi_m, Pn) and in the hot-air temperature u, which may be a scalar
(constant drive, the paper's setting, u=1) or a length-(n_t-1) sequence (the control setting).
"""
from __future__ import annotations

import torch

DTYPE = torch.float64


def _d2_operator(n_x: int, h: float, bi_left: torch.Tensor):
    """Dense second-derivative operator with Robin BC at xi=0 and Neumann at xi=1, plus BC source.

    Mirrors fd_solver._build_d2_operator exactly (ghost-point elimination):
      row 0    : diag = -2(1 + h*bi)/h^2,  sup = 2/h^2,  source g[0] = 2*bi/h
      interior : [1, -2, 1] / h^2
      row n-1  : diag = -2/h^2, sub = 2/h^2   (insulated)

    Built so that `bi_left` stays in the autograd graph.
    """
    inv = 1.0 / h ** 2
    L = torch.zeros(n_x, n_x, dtype=DTYPE)
    idx = torch.arange(1, n_x - 1)
    L[idx, idx - 1] = inv
    L[idx, idx] = -2.0 * inv
    L[idx, idx + 1] = inv
    L[n_x - 1, n_x - 1] = -2.0 * inv
    L[n_x - 1, n_x - 2] = 2.0 * inv
    L[0, 1] = 2.0 * inv

    # row-0 diagonal and the BC source are the only bi-dependent entries; add them as
    # rank-one / vector terms so the graph carries bi_left.
    e0 = torch.zeros(n_x, dtype=DTYPE)
    e0[0] = 1.0
    L = L + torch.outer(e0, e0) * (-2.0 * (1.0 + h * bi_left) * inv)
    g = e0 * (2.0 * bi_left / h)
    return L, g


def solve_luikov_torch(Lu, Ko, Bi_t, Bi_m, Pn=0.0, n_x: int = 101, n_t: int = 201,
                       tau_final: float = 2.0, u=None):
    """Return (T, M) of shape (n_t, n_x), differentiable in the parameters and in u.

    u : None or scalar -> constant hot-air temperature (default 1.0, the paper's convention).
        A length-(n_t-1) tensor gives a time-varying drive (the control setting).
    """
    to = lambda v: v if torch.is_tensor(v) else torch.tensor(float(v), dtype=DTYPE)
    Lu, Ko, Bi_t, Bi_m, Pn = map(to, (Lu, Ko, Bi_t, Bi_m, Pn))

    h = 1.0 / (n_x - 1)
    dtau = tau_final / (n_t - 1)

    L_T, g_T_unit = _d2_operator(n_x, h, Bi_t)
    L_M, g_M_unit = _d2_operator(n_x, h, Bi_m)
    L_M = Lu * L_M                      # moisture diffusivity scaling (as in the reference)
    g_M = Lu * g_M_unit * 0.0           # M_inf = 0

    eye = torch.eye(n_x, dtype=DTYPE)
    A_T = torch.linalg.lu_factor(eye - dtau * L_T)
    A_M = torch.linalg.lu_factor(eye - dtau * L_M)

    if u is None:
        u_seq = torch.ones(n_t - 1, dtype=DTYPE)
    elif torch.is_tensor(u) and u.ndim == 1:
        u_seq = u
    else:
        u_seq = to(u).expand(n_t - 1)

    T = torch.zeros(n_x, dtype=DTYPE)
    M = torch.ones(n_x, dtype=DTYPE)
    Ts, Ms = [T], [M]
    for n in range(n_t - 1):
        g_T = g_T_unit * u_seq[n]
        rhs_M = M + dtau * g_M
        if float(Pn.detach()) != 0.0 or Pn.requires_grad:
            d2T = L_T @ T + g_T
            rhs_M = rhs_M + dtau * Lu * Pn * d2T
        M_new = torch.linalg.lu_solve(*A_M, rhs_M.unsqueeze(-1)).squeeze(-1)
        dMdtau = (M_new - M) / dtau
        rhs_T = T + dtau * g_T + dtau * Ko * dMdtau
        T_new = torch.linalg.lu_solve(*A_T, rhs_T.unsqueeze(-1)).squeeze(-1)
        T, M = T_new, M_new
        Ts.append(T); Ms.append(M)
    return torch.stack(Ts), torch.stack(Ms)


def channels_torch(T, M, Lu, Bi_t, Bi_m, Pn, K: int = 40, with_T: bool = False, u=None):
    """Shared observation operator in torch: [Mbar, q_m(0,tau)] (+ [Tbar]).

    Mirrors theory/observe.channels, including the conservation-form surface flux
        q_m(0,tau) = -Lu ( Bi_m M(0,tau) + Pn Bi_t (T(0,tau) - u) )
    which equals d<M>/dtau exactly. u defaults to 1 (the paper's constant drive).
    """
    n_t = T.shape[0]
    idx = torch.linspace(1, n_t - 1, K).round().long()
    u_t = torch.ones(n_t, dtype=DTYPE) if u is None else (
        u if torch.is_tensor(u) and u.ndim == 1 else torch.as_tensor(u, dtype=DTYPE).expand(n_t))
    if u_t.shape[0] == n_t - 1:                       # control sequences are one shorter
        u_t = torch.cat([u_t[:1], u_t])
    Mbar = M.mean(dim=1)[idx]
    q = -Lu * (Bi_m * M[:, 0] + Pn * Bi_t * (T[:, 0] - u_t))
    out = [Mbar, q[idx]]
    if with_T:
        out.append(T.mean(dim=1)[idx])
    return torch.cat(out)
