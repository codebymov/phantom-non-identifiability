"""E-optimal sensor design for the Luikov drying cell (the prescriptive turn).

Proposition 1 (frozen paper, App. A) says a moisture-only observation operator leaves
{Ko, Bi_t} on a two-dimensional Fisher null space. This module turns that *diagnosis* into a
*prescription*: given a library of candidate observation functionals, which subset (under a sensor
budget) makes the parameter vector observable?

We reuse the SAME scaled-sensitivity-through-the-solver method as theory/identifiability_fim.py
(central differences, theta = (Lu, Ko, Bi_t, Bi_m) = (0.5, 2, 1, 1), K = 40 sample times, NOISE =
0.01) so every number here is consistent with the frozen paper's Appendix A. The two functions
observe_channels / sensitivity_block are COPIES of that file's method, generalized to an arbitrary
candidate list (the original only supports the two hardcoded sets "moist" / "moist+temp"); the
original file is not modified.

Selection objective:
  E-optimality  = maximize  lambda_min(F),  F = S^T S / sigma^2   (worst-resolved direction)
  A-optimality  = minimize  trace(F^{-1})   = sum of Cramer-Rao variances (secondary)

Run:  PYTHONPATH=src .venv/bin/python theory/sensor_design.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from kibapap.physics import LuikovParams, solve_luikov_fd

# --- setup identical to theory/identifiability_fim.py (keeps numbers consistent with App. A) ---
NAMES = ["Lu", "Ko", "Bi_t", "Bi_m"]
THETA0 = np.array([0.5, 2.0, 1.0, 1.0])
K = 40
NOISE = 0.01
N_X, N_T = 101, 201
BUDGET = 3
OUT = Path(__file__).resolve().parents[1] / "theory/sensor_design.json"

# Candidate observation functionals. Each maps an FD solution to a length-K row-block.
# "family" marks moisture vs temperature (the decoy must be moisture-only).
MID = N_X // 2  # xi = 0.5 index for n_x = 101 -> 50


def _idx(tau):
    return np.linspace(1, len(tau) - 1, K).astype(int)


CANDIDATES = {
    "M_mean":    dict(family="moisture", label=r"$\langle M\rangle$ (mean moisture)",
                      fn=lambda s: s.M.mean(1)[_idx(s.tau)]),
    "M_surfrate": dict(family="moisture", label=r"$\partial_\tau M(0,\tau)$ (surface rate)",
                      fn=lambda s: np.gradient(s.M[:, 0], s.tau)[_idx(s.tau)]),
    "M_mid":     dict(family="moisture", label=r"$M(0.5,\tau)$ (mid-depth moisture, decoy)",
                      fn=lambda s: s.M[:, MID][_idx(s.tau)]),
    "T_mean":    dict(family="temperature", label=r"$\langle T\rangle$ (mean temperature)",
                      fn=lambda s: s.T.mean(1)[_idx(s.tau)]),
    "T_surf":    dict(family="temperature", label=r"$T(0,\tau)$ (surface temperature)",
                      fn=lambda s: s.T[:, 0][_idx(s.tau)]),
}


def _solve(theta):
    return solve_luikov_fd(LuikovParams(*theta), n_x=N_X, n_t=N_T)


def sensitivity_blocks(theta=THETA0, d=1e-3):
    """Scaled sensitivity S_ij = d f_i / d log theta_j per candidate, by central differences.

    Copy of identifiability_fim.py's method, generalized to all candidates at once: 8 FD solves
    (4 params x +/-) feed every candidate.

    Args:
        theta: Parameter point to linearise about. Defaults to ``THETA0``.
        d: Relative central-difference step in log-parameter space.

    Returns:
        dict: Candidate name -> its ``(K, 4)`` sensitivity block.
    """
    per_param = []
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] *= (1 + d); tm[j] *= (1 - d)
        sp, sm = _solve(tp), _solve(tm)
        per_param.append((sp, sm))
    blocks = {}
    for name, c in CANDIDATES.items():
        cols = [(c["fn"](sp) - c["fn"](sm)) / (2 * d) for (sp, sm) in per_param]
        blocks[name] = np.array(cols).T  # (K, 4)
    return blocks


def fisher_metrics(S):
    """Design criteria for one candidate sensor set.

    Args:
        S: Stacked sensitivity block for the chosen set.

    Returns:
        dict: ``lam_min`` (E-optimality), ``a_opt`` (A-optimality, ``trace(F^-1)``), ``rank``,
            the eigenvalues in descending order, and the per-parameter Cramer-Rao relative
            standard deviation.
    """
    F = S.T @ S / NOISE ** 2
    w = np.linalg.eigvalsh(F)                       # ascending, real (F symmetric PSD)
    w = np.clip(w, 0, None)
    lam_min = float(w[0])
    tol = 1e-9 * max(w.max(), 1.0)
    rank = int((w > tol).sum())
    crb = np.full(len(NAMES), np.inf)
    a_opt = np.inf
    if rank == len(NAMES):
        Finv = np.linalg.inv(F)
        crb = np.sqrt(np.diag(Finv))
        a_opt = float(np.trace(Finv))
    return dict(lam_min=lam_min, a_opt=a_opt, rank=rank, eig=w[::-1].tolist(),
                crb=dict(zip(NAMES, crb.tolist())))


def score_subset(blocks, names):
    S = np.vstack([blocks[n] for n in names])
    m = fisher_metrics(S)
    m["set"] = list(names)
    m["families"] = sorted({CANDIDATES[n]["family"] for n in names})
    return m


def main():
    blocks = sensitivity_blocks()
    # Always include the two current sensors (M_mean, M_surfrate); choose the rest under budget.
    fixed = ["M_mean", "M_surfrate"]
    optional = [n for n in CANDIDATES if n not in fixed]

    subsets = []
    for k in range(0, BUDGET - len(fixed) + 1):          # add 0..(B-2) extra channels
        for extra in combinations(optional, k):
            subsets.append(score_subset(blocks, fixed + list(extra)))
    subsets.sort(key=lambda m: (-m["lam_min"], m["a_opt"]))

    print(f"theta = {dict(zip(NAMES, THETA0))}, K={K}, noise={NOISE}, budget B={BUDGET}\n")
    print(f"  {'sensor set':52s} {'rank':>4s} {'lam_min':>10s} {'A-opt':>10s}")
    for m in subsets:
        a = f"{m['a_opt']:.3g}" if np.isfinite(m["a_opt"]) else "inf"
        print(f"  {'+'.join(m['set']):52s} {m['rank']:>4d} {m['lam_min']:>10.4g} {a:>10s}")

    # S_star: best under budget (max lam_min).  S_decoy: the plausible "more data" moisture-only
    # choice -- spend the budget on an extra moisture sensor (most channels), which still fails.
    S_star = subsets[0]
    moisture_only = [m for m in subsets if m["families"] == ["moisture"]]
    moisture_only.sort(key=lambda m: (-len(m["set"]), -m["lam_min"]))
    S_decoy = moisture_only[0]

    print(f"\n  S_star  (E-optimal pick) = {'+'.join(S_star['set'])}"
          f"   lam_min={S_star['lam_min']:.4g}  rank={S_star['rank']}")
    print(f"  S_decoy (best moisture-only) = {'+'.join(S_decoy['set'])}"
          f"   lam_min={S_decoy['lam_min']:.4g}  rank={S_decoy['rank']}")

    # --- Decoy/pick asserts: the whole result hinges on these. STOP if they fail. ---
    assert S_decoy["lam_min"] < 1e-6, f"decoy lam_min not ~0: {S_decoy['lam_min']}"
    assert S_decoy["rank"] == 2, f"decoy rank not 2: {S_decoy['rank']}"
    assert S_star["lam_min"] > 1e-6, f"pick lam_min not > 0: {S_star['lam_min']}"
    assert S_star["rank"] == 4, f"pick rank not 4: {S_star['rank']}"
    print("\n  ASSERTS OK: decoy is rank-2 lam_min=0 (Ko,Bi_t unrecoverable); "
          "pick is rank-4 lam_min>0.")

    OUT.write_text(json.dumps({
        "theta0": dict(zip(NAMES, THETA0.tolist())), "K": K, "noise": NOISE, "budget": BUDGET,
        "candidates": {n: {"family": c["family"]} for n, c in CANDIDATES.items()},
        "ranked": subsets, "S_star": S_star, "S_decoy": S_decoy,
    }, indent=2))
    print(f"\nsaved {OUT}")
    return S_star, S_decoy


if __name__ == "__main__":
    main()
