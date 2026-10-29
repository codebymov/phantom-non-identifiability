"""Tangent maps of the observation operator: J_theta = dH/dtheta and J_u = dH/du.

Three interfaces:

  jacobian_ad(...)  exact, by reverse-mode AD through the differentiable mirror solver;
  jacobian_fd(...)  the central-difference oracle the predecessor paper used, kept as a
                    *verification* path rather than the production one;
  jvp(...)          Jacobian-vector products, which is what makes derivative supervision
                    affordable at scale (one solve per direction rather than one per parameter).

AD-vs-FD agreement is a standing test (tangentsafe/tests/), following the pattern the predecessor
established with the two-head control: an independent oracle for a derivative that the whole argument
rests on.

Sign/scaling convention matches theory/observe.py: columns are scaled sensitivities
    S_ij = dH_i / dlog theta_j,
so entries are dimensionless and comparable across parameters.
"""
from __future__ import annotations

import numpy as np
import torch

from tangentsafe.core.solvers.luikov_torch import DTYPE, channels_torch, solve_luikov_torch

THETA_NAMES = ["Lu", "Ko", "Bi_t", "Bi_m", "Pn"]
NOMINAL = dict(Lu=0.5, Ko=2.0, Bi_t=1.0, Bi_m=1.0, Pn=0.0)


def _observe(theta_vec, K=40, with_T=False, n_x=101, n_t=201, u=None):
    Lu, Ko, Bi_t, Bi_m, Pn = theta_vec
    T, M = solve_luikov_torch(Lu, Ko, Bi_t, Bi_m, Pn, n_x=n_x, n_t=n_t, u=u)
    return channels_torch(T, M, Lu, Bi_t, Bi_m, Pn, K=K, with_T=with_T, u=u)


def observe(theta: dict, **kw):
    v = [torch.tensor(float(theta[n]), dtype=DTYPE) for n in THETA_NAMES]
    return _observe(v, **kw)


def jacobian_ad(theta: dict, params=("Lu", "Ko", "Bi_t", "Bi_m"), K=40, with_T=False,
                n_x=101, n_t=201, u=None) -> np.ndarray:
    """Scaled Jacobian dH/dlog theta_j by reverse-mode AD. Shape (n_obs, len(params))."""
    vals = {n: torch.tensor(float(theta[n]), dtype=DTYPE, requires_grad=(n in params))
            for n in THETA_NAMES}
    H = _observe([vals[n] for n in THETA_NAMES], K=K, with_T=with_T, n_x=n_x, n_t=n_t, u=u)
    cols = []
    for j, name in enumerate(params):
        grads = torch.autograd.grad(H, vals[name], grad_outputs=torch.eye(H.numel(), dtype=DTYPE),
                                    retain_graph=True, is_grads_batched=True)[0]
        cols.append(grads.detach().numpy() * float(theta[name]))   # chain rule to d/dlog
    return np.stack(cols, axis=1)


def jacobian_fd(theta: dict, params=("Lu", "Ko", "Bi_t", "Bi_m"), d=1e-3, K=40, with_T=False,
                n_x=101, n_t=201, u=None) -> np.ndarray:
    """Central-difference oracle in log-parameter space (the predecessor's method)."""
    cols = []
    for name in params:
        tp, tm = dict(theta), dict(theta)
        tp[name] = theta[name] * (1 + d); tm[name] = theta[name] * (1 - d)
        hp = observe(tp, K=K, with_T=with_T, n_x=n_x, n_t=n_t, u=u).detach().numpy()
        hm = observe(tm, K=K, with_T=with_T, n_x=n_x, n_t=n_t, u=u).detach().numpy()
        cols.append((hp - hm) / (2 * d))
    return np.stack(cols, axis=1)


def jvp(theta: dict, direction: dict, K=40, with_T=False, n_x=101, n_t=201, u=None) -> np.ndarray:
    """Jacobian-vector product in log-parameter space: one forward-mode solve per direction."""
    prim = tuple(torch.tensor(float(theta[n]), dtype=DTYPE) for n in THETA_NAMES)
    tang = tuple(torch.tensor(float(direction.get(n, 0.0)) * float(theta[n]), dtype=DTYPE)
                 for n in THETA_NAMES)
    _, out = torch.autograd.functional.jvp(
        lambda *v: _observe(v, K=K, with_T=with_T, n_x=n_x, n_t=n_t, u=u), prim, tang)
    return out.detach().numpy()


def jacobian_u_ad(theta: dict, u_val=1.0, K=40, with_T=False, n_x=101, n_t=201) -> np.ndarray:
    """Input-to-observable tangent dH/du for a constant hot-air drive u."""
    u = torch.tensor(float(u_val), dtype=DTYPE, requires_grad=True)
    vals = [torch.tensor(float(theta[n]), dtype=DTYPE) for n in THETA_NAMES]
    H = _observe(vals, K=K, with_T=with_T, n_x=n_x, n_t=n_t, u=u)
    g = torch.autograd.grad(H, u, grad_outputs=torch.eye(H.numel(), dtype=DTYPE),
                            is_grads_batched=True)[0]
    return g.detach().numpy().reshape(-1, 1)
