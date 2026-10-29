"""Section-8 metrics: how wrong is a surrogate's tangent map, in the ways that change decisions.

Forward L2 is reported alongside these precisely to show it is uninformative about them.

The metric that matters for design is the sensitive-subspace angle: the E-optimal sensor verdict
depends on the dominant eigendirections of the Fisher matrix, not on entrywise Jacobian accuracy. A
surrogate can have small ||J_sur - J_ref||_F and still rank sensors differently.
"""
from __future__ import annotations

import numpy as np


def fisher_from_jacobian(J: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
    """Fisher information from a Jacobian and a noise model.

    Args:
        J: Jacobian of shape ``(n_obs, n_params)``.
        sigma: Per-observable noise standard deviation, or one scalar for all.

    Returns:
        ``F = J^T Sigma^-1 J``, shape ``(n_params, n_params)``.
    """
    w = np.asarray(sigma, dtype=float)
    Jw = J / (w[:, None] if w.ndim else w)
    return Jw.T @ Jw


def subspace_angle(A: np.ndarray, B: np.ndarray, k: int = 2) -> float:
    """Largest principal angle between the top-k right singular subspaces of two matrices.

    Args:
        A: First matrix.
        B: Second matrix.
        k: Number of leading right singular vectors to compare.

    Returns:
        float: The angle in radians. ``pi/2`` when either subspace is degenerate, for instance
            against a structurally zero Jacobian -- which is the correct answer, since nothing
            about the reference orientation is recovered.
    """
    def top_right(M):
        _U, s, Vt = np.linalg.svd(M, full_matrices=False)
        r = int((s > 1e-12 * max(s.max(), 1e-300)).sum())
        if r == 0:
            return None
        return Vt[: min(k, r)].T
    Va, Vb = top_right(A), top_right(B)
    if Va is None or Vb is None or Va.shape[1] != Vb.shape[1]:
        return float(np.pi / 2)
    s = np.linalg.svd(Va.T @ Vb, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


def _cr(J, sigma, names):
    F = fisher_from_jacobian(J, sigma)
    cn = np.linalg.norm(J / (np.asarray(sigma)[:, None] if np.ndim(sigma) else sigma), axis=0)
    rank = int((cn > 1e-9 * cn.max()).sum()) if cn.max() > 0 else 0
    crb = np.full(len(names), np.inf)
    if rank == len(names):
        try:
            d = np.diag(np.linalg.solve(F, np.eye(len(names))))
            if np.all(np.isfinite(d)) and np.all(d > 0):
                crb = np.sqrt(d) * 100
        except np.linalg.LinAlgError:
            pass
    return rank, crb, cn


def tangent_metrics(J_sur: np.ndarray, J_ref: np.ndarray, sigma, names, k: int = 2,
                    fwd_relL2: float | None = None) -> dict:
    """Compare one surrogate or reduction against a reference tangent map.

    Args:
        J_sur: Surrogate or reduced-model Jacobian.
        J_ref: Reference Jacobian.
        sigma: Noise model passed to `fisher_from_jacobian`.
        names: Parameter names, for labelling the per-parameter outputs.
        k: Subspace dimension for the principal-angle comparison.
        fwd_relL2: Forward relative L2 error, recorded alongside so forward accuracy and
            tangent accuracy can be read together. Defaults to ``None``.

    Returns:
        dict: Relative Jacobian error, singular-value error, subspace angle in radians and
            degrees, the rank of each Jacobian, the Cramer-Rao bound from each, and the
            sensitivity ordering each implies -- the block the paper reports per coupling.
    """
    nref = np.linalg.norm(J_ref)
    E_J = float(np.linalg.norm(J_sur - J_ref) / nref) if nref > 0 else float("nan")
    s_sur = np.linalg.svd(J_sur, compute_uv=False)
    s_ref = np.linalg.svd(J_ref, compute_uv=False)
    m = min(len(s_sur), len(s_ref))
    E_sig = float(np.linalg.norm(s_sur[:m] - s_ref[:m]) / max(np.linalg.norm(s_ref[:m]), 1e-300))
    rank_s, crb_s, cn_s = _cr(J_sur, sigma, names)
    rank_r, crb_r, cn_r = _cr(J_ref, sigma, names)

    # decision-level: does the E-optimal ranking of "which parameter is worst resolved" agree?
    order_s = [names[i] for i in np.argsort(-cn_s)]
    order_r = [names[i] for i in np.argsort(-cn_r)]
    return dict(
        E_J=E_J, E_sigma=E_sig,
        subspace_angle_rad=subspace_angle(J_sur, J_ref, k),
        subspace_angle_deg=float(np.degrees(subspace_angle(J_sur, J_ref, k))),
        rank_surrogate=rank_s, rank_reference=rank_r,
        cr_surrogate=dict(zip(names, [None if not np.isfinite(v) else float(v) for v in crb_s])),
        cr_reference=dict(zip(names, [None if not np.isfinite(v) else float(v) for v in crb_r])),
        col_norm_surrogate=dict(zip(names, cn_s.tolist())),
        col_norm_reference=dict(zip(names, cn_r.tolist())),
        sensitivity_order_surrogate=order_s, sensitivity_order_reference=order_r,
        order_agrees=bool(order_s == order_r),
        forward_relL2=fwd_relL2,
    )
