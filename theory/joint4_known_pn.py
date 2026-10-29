"""Joint 4-parameter recovery from the two moisture channels, with Pn known and fixed.

The reviews asked the obvious next question. Table 2 recovers (Ko, Bi_t) with Lu and Bi_m held at
truth, and App. B's five-parameter run (which also estimates Pn) is multimodal. Neither answers:
with the coupling known, can a moisture-only cell recover ALL FOUR of (Lu, Ko, Bi_t, Bi_m) at once?

That is the case a practitioner is in after running the audit of App. D: the dropped term has been
restored and its coefficient measured, so Pn is known, and what remains is the four-parameter
inverse problem on the coupled model.

Setup matches Table 2's estimator: solver-in-the-loop Levenberg-Marquardt-style least squares in
log-parameter space (scipy trf, bounded to the operating box), the shared observation model of
theory/observe.py (mean moisture + surface mass flux, independent per-channel noise), 20 noise
seeds, and N_STARTS random initialisations drawn from the box per seed. The reported estimate per
seed is the lowest-cost start, which is what multistart is for.

Reported per (Pn, noise) cell:
  - median relative error per parameter, over seeds, of the best-cost estimate;
  - frac_basin: fraction of all starts whose converged cost is within 5% of the best cost found for
    that seed -- i.e. how often a random start reaches the best basin at all;
  - frac_recovered: fraction of seeds whose best-cost estimate is within REC_TOL of truth on all
    four parameters. This is the honest check, since reaching a consensus basin is not the same as
    reaching the right one;
  - cr_marginal: the marginal Cramer-Rao bound from the same Fisher analysis as Table 1, scaled to
    the cell's noise level (CR is linear in sigma), for comparison with the achieved error.

Run:  PYTHONPATH=. .venv/bin/python theory/joint4_known_pn.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from theory.observe import FRAC, K_DEFAULT, NAMES4, channels, fisher, sample_times, solve

THETA_TRUE = [0.5, 2.0, 1.0, 1.0]            # Lu, Ko, Bi_t, Bi_m
BOX_LO = [0.2, 1.0, 0.5, 0.5]
BOX_HI = [1.0, 3.0, 2.0, 2.0]
PN_GRID = [0.2, 0.6, 1.0]
FRACS = [0.005, 0.01, 0.02]
SEEDS = list(range(20))
N_STARTS = 4
MAX_NFEV = 600
BASIN_TOL = 0.05                             # cost within 5% of the seed's best counts as the basin
REC_TOL = 0.10                               # all four within 10% of truth counts as recovered
OUT = Path(__file__).resolve().parents[1] / "runs/theory/joint4_known_pn.json"

LO, HI = np.log(BOX_LO), np.log(BOX_HI)


def fit_cell(pn, frac, rng_starts):
    """Recover all four parameters at one (Pn, noise) cell, with Pn known and fixed.

    Args:
        pn (float): The true coupling, also given to the estimator's forward model.
        frac (float): Noise standard deviation as a fraction of each channel's range.
        rng_starts (numpy.random.Generator): Generator for the random initialisations. Shared
            across cells so the start sequence is reproducible for the whole sweep.

    Returns:
        dict: With keys ``median_err_pct`` and ``median_est`` (per parameter, over seeds, of
            the lowest-cost start), ``frac_basin`` (fraction of all starts landing within
            `BASIN_TOL` of their seed's best cost) and ``frac_recovered`` (fraction of seeds
            whose best estimate is within `REC_TOL` of truth on all four parameters at once).
    """
    truth = solve(THETA_TRUE, pn)
    ts = sample_times(truth, K_DEFAULT)
    ys = channels(truth, ts)
    sig = [frac * (c.max() - c.min()) for c in ys]

    def resid(logp, obs):
        try:
            pred = channels(solve(np.exp(np.clip(logp, LO, HI)), pn), ts)
        except (ValueError, FloatingPointError):
            return np.full(2 * K_DEFAULT, 1e3)
        r = np.concatenate([(p - o) / s for p, o, s in zip(pred, obs, sig)])
        return r if np.all(np.isfinite(r)) else np.full(2 * K_DEFAULT, 1e3)

    best, in_basin = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(5000 + sd)
        obs = [y + rng.normal(0, s, y.shape) for y, s in zip(ys, sig)]
        ests, costs = [], []
        for _ in range(N_STARTS):
            x0 = rng_starts.uniform(LO, HI)
            r = least_squares(resid, x0, args=(obs,), method="trf",
                              bounds=(LO, HI), max_nfev=MAX_NFEV)
            ests.append(np.exp(r.x)); costs.append(float(r.cost))
        costs = np.array(costs); k = int(np.argmin(costs))
        best.append(ests[k])
        in_basin.extend(costs <= (1.0 + BASIN_TOL) * costs[k])
    best = np.array(best)
    rel = np.abs(best - np.array(THETA_TRUE)) / np.array(THETA_TRUE)
    return dict(
        median_err_pct={n: float(np.median(rel[:, i]) * 100) for i, n in enumerate(NAMES4)},
        median_est={n: float(np.median(best[:, i])) for i, n in enumerate(NAMES4)},
        frac_basin=float(np.mean(in_basin)),
        frac_recovered=float(np.mean(np.all(rel <= REC_TOL, axis=1))),
    )


def main() -> None:
    rng_starts = np.random.default_rng(11)
    cells = []
    print(f"joint 4-parameter recovery, Pn known; {len(SEEDS)} seeds x {N_STARTS} starts per cell")
    print(f"truth {dict(zip(NAMES4, THETA_TRUE))}, box {BOX_LO} to {BOX_HI}\n")
    hdr = (f"{'Pn':>5} {'noise':>6} | " + " ".join(f"{n+' err':>10}" for n in NAMES4)
           + f" | {'basin':>6} {'recov':>6}")
    print(hdr); print("-" * len(hdr))
    for pn in PN_GRID:
        cr1 = fisher(THETA_TRUE, pn)["crb"]          # marginal CR at FRAC; CR is linear in sigma
        for frac in FRACS:
            c = fit_cell(pn, frac, rng_starts)
            c["pn"] = pn; c["noise_frac"] = frac
            c["cr_marginal_pct"] = {n: float(cr1[i] * frac / FRAC) for i, n in enumerate(NAMES4)}
            cells.append(c)
            print(f"{pn:5.2f} {frac*100:5.1f}% | "
                  + " ".join(f"{c['median_err_pct'][n]:9.2f}%" for n in NAMES4)
                  + f" | {c['frac_basin']*100:5.0f}% {c['frac_recovered']*100:5.0f}%")
        print(f"{'':5} {'CR  ':>6} | "
              + " ".join(f"{cr1[i]:9.2f}%" for i in range(4)) + "   (marginal, at 1% noise)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        theta_true=dict(zip(NAMES4, THETA_TRUE)), box_lo=BOX_LO, box_hi=BOX_HI,
        pn_grid=PN_GRID, noise_fracs=FRACS, seeds=len(SEEDS), n_starts=N_STARTS,
        max_nfev=MAX_NFEV, basin_tol=BASIN_TOL, rec_tol=REC_TOL, cells=cells), indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
