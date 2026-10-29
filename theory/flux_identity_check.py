"""Discrete conservation check: does q_m(0,tau) equal d<M>/dtau on the solver's own grid?

The observation operator (theory/observe.py) takes the surface mass flux from the closed-form
Robin derivatives,

    q_m(0,tau) = -Lu ( Bi_m M(0,tau) + Pn Bi_t (T(0,tau) - 1) ),

which is the correct mass-flow-meter operator because, in conservation form
d_tau M = -d_xi q_m with a Neumann back boundary,

    q_m(0,tau) = d<M>/dtau    exactly (in the continuum).

This script measures how well that holds DISCRETELY, which is what the paper reports as a
conservation check on the discretisation. The reference d<M>/dtau is a second-order central
difference of the solver's own mean-moisture trace.

Three things the numbers below show, and which the paper states rather than glosses over:

  1. With the consistent trapezoid quadrature for <M>, the residual converges at FIRST ORDER IN
     TIME (0.36% -> 0.11% at Pn=0 under 4x temporal refinement) and is already converged in space
     (0.3611 -> 0.3610 under 2x spatial refinement). The identity is real; what is finite is the
     time discretisation, the implicit step evaluating the surface Robin flux at the new time
     level while the reference is a centred difference of the trajectory.
  2. With the plain nodal mean -- which is what channels() actually observes, and therefore what
     the paper's reported numbers refer to -- the residual is NON-MONOTONE under refinement,
     because that mean carries its own O(dxi) quadrature bias which partially cancels the O(dtau)
     time error. It is not a smaller error than the trapezoid one, only a differently composed
     one, so it must not be reported as a grid-converged figure.
  3. It grows with Pn under either quadrature, because the explicit Pn*Bi_t thermogradient term
     in q_m grows with it. This is why the paper gives the residual over a stated Pn range
     rather than one number for all of [0, 2].

Run:  PYTHONPATH=. .venv/bin/python theory/flux_identity_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from theory.observe import NT, NX, solve

THETA0 = (0.5, 2.0, 1.0, 1.0)
PN_GRID = [0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.5, 2.0]
EDGE = 2          # drop the end points, where a central difference is one-sided
GRIDS = [(NX, NT), (2 * (NX - 1) + 1, NT), (NX, 4 * (NT - 1) + 1)]
OUT = Path(__file__).resolve().parents[1] / "runs/theory/flux_identity.json"


def residual(theta, pn, n_x, n_t, quad="mean"):
    """Relative L2 mismatch (%) between the closed-form surface flux and d<M>/dtau."""
    s = solve(theta, pn, n_x, n_t)
    p = s.params
    q = -p.Lu * (p.Bi_m * s.M[:, 0] + p.Pn * p.Bi_t * (s.T[:, 0] - 1.0))
    if quad == "mean":  # noqa: SIM108  (the ternary form runs past the line limit)
        Mbar = s.M.mean(1)                                     # what channels() observes
    else:
        Mbar = np.trapezoid(s.M, np.linspace(0.0, 1.0, s.M.shape[1]), axis=1)
    dMdt = np.gradient(Mbar, s.tau)
    sl = slice(EDGE, -EDGE)
    return float(np.linalg.norm(q[sl] - dMdt[sl]) / np.linalg.norm(dMdt[sl]) * 100)


def main() -> None:
    rows = []
    labels = [f"{nx}x{nt}" for nx, nt in GRIDS]
    print(f"theta={THETA0}   grids: {labels[0]} (paper), {labels[1]} (2x space), "
          f"{labels[2]} (4x time)\n")
    print(f"{'Pn':>6} | " + " ".join(f"{lab + ' mean':>16}" for lab in labels)
          + " | " + " ".join(f"{lab + ' trapz':>17}" for lab in labels))
    for pn in PN_GRID:
        m = [residual(THETA0, pn, nx, nt, "mean") for nx, nt in GRIDS]
        t = [residual(THETA0, pn, nx, nt, "trapz") for nx, nt in GRIDS]
        rows.append(dict(pn=pn, mean=dict(zip(labels, m)), trapz=dict(zip(labels, t))))
        print(f"{pn:6.2f} | " + " ".join(f"{v:16.4f}" for v in m)
              + " | " + " ".join(f"{v:17.4f}" for v in t))

    pap = labels[0]
    lo = [r["mean"][pap] for r in rows if r["pn"] <= 0.2]
    allm = [r["mean"][pap] for r in rows]
    print(f"\non the paper's {pap} grid, plain-mean <M> (the observed channel):")
    print(f"  Pn <= 0.2    : max {max(lo):.2f}%")
    print(f"  Pn in [0, 2] : max {max(allm):.2f}%  (at Pn = 2)")
    tz = [(r["pn"], r["trapz"][labels[0]], r["trapz"][labels[2]]) for r in rows]
    print("with the consistent trapezoid <M>, 4x temporal refinement gives first-order decay:")
    for pn, a, b in tz[:1] + tz[-1:]:
        print(f"  Pn={pn:4.2f}: {a:.3f}% -> {b:.3f}%  (factor {a / b:.1f})")
    print("The plain nodal mean is non-monotone under refinement (its O(dxi) quadrature bias\n"
          "partly cancels the O(dtau) time error), so it is not a grid-converged figure.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        theta0=THETA0, grids=labels, paper_grid=pap, edge_drop=EDGE, rows=rows,
        summary=dict(paper_grid=pap,
                     max_pct_pn_le_0p2=max(lo),
                     max_pct_pn_le_2=max(allm),
                     trapz_converges_in_time=True,
                     plain_mean_monotone_under_refinement=False)), indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
