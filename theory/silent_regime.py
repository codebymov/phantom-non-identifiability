"""Is the phantom silent? Forward goodness-of-fit vs. the identifiability verdict.

The obvious objection to the phantom-identifiability result is that a practitioner would NOTICE
that the reduced model is wrong. This script tests the objection under the shared observation model
(mean moisture + surface mass flux, independent per-channel noise at 1% of range).

For each true Posnov number we generate noisy moisture-only observations from the FULL model and
then FIT THE REDUCED MODEL over the parameters it can actually see, (Lu, Bi_m) -- the generous
version of the practitioner, who calibrates before judging. We then ask whether the calibrated
reduced model's residual is distinguishable from sensor noise, via the chi-square goodness-of-fit
test at the known noise level:

    chi2 = sum_i (r_i / sigma_i)^2  ~  chi2(2K - p)   under a correctly specified model.

If it PASSES (p > alpha) the misspecification is invisible, while the same reduced model still
reports rank 2, CR = infinity and "buy a temperature sensor" -- the phantom is SILENT.

Run:  PYTHONPATH=. .venv/bin/python theory/silent_regime.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from theory.observe import FRAC, K_DEFAULT, channels, sample_times, sigmas, solve

THETA_TRUE = [0.5, 2.0, 1.0, 1.0]
SEEDS = list(range(20))
ALPHA = 0.05
PN_GRID = np.concatenate([[0.0], np.geomspace(0.005, 2.0, 25)])
OUT = Path(__file__).resolve().parents[1] / "runs/theory/silent_regime.json"


def main() -> None:
    rows = []
    # The calibration is bounded to a physically plausible box: a practitioner would reject a fit
    # that ran Lu off to 1e4 rather than report it as a passing model.
    for pn_true in PN_GRID:
        truth = solve(THETA_TRUE, pn_true)
        ts = sample_times(truth, K_DEFAULT)
        ys = channels(truth, ts)
        sig = sigmas(ys)

        red_nom = solve(THETA_TRUE, 0.0)
        relL2_nom = float(np.linalg.norm(red_nom.M - truth.M) / np.linalg.norm(truth.M))

        def resid(logp, obs):
            Lu, Bim = np.exp(logp)
            pred = channels(solve([Lu, THETA_TRUE[1], THETA_TRUE[2], Bim], 0.0), ts)
            return np.concatenate([(p - o) / s for p, o, s in zip(pred, obs, sig)])

        ps, Lus, Bims, rel = [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(1000 + sd)
            obs = [y + rng.normal(0, s, y.shape) for y, s in zip(ys, sig)]
            r = least_squares(resid, np.log([THETA_TRUE[0], THETA_TRUE[3]]), args=(obs,),
                              method="trf", bounds=(np.log([0.05, 0.1]), np.log([2.0, 10.0])),
                              max_nfev=300)
            Lu_f, Bim_f = np.exp(r.x)
            dof = 2 * K_DEFAULT - 2
            ps.append(float(chi2_dist.sf(float(np.sum(r.fun ** 2)), dof)))
            Lus.append(Lu_f); Bims.append(Bim_f)
            s = solve([Lu_f, THETA_TRUE[1], THETA_TRUE[2], Bim_f], 0.0)
            rel.append(float(np.linalg.norm(s.M - truth.M) / np.linalg.norm(truth.M)))

        frac_pass = float(np.mean(np.array(ps) > ALPHA))
        rows.append(dict(pn=float(pn_true), p=float(np.median(ps)), frac_pass=frac_pass,
                         Lu_fit=float(np.mean(Lus)), Bim_fit=float(np.mean(Bims)),
                         relL2_calibrated=float(np.mean(rel)), relL2_nominal=relL2_nom))
        print(f"Pn={pn_true:6.4f} | p={np.median(ps):7.4f} pass={frac_pass:4.2f} | "
              f"Lu_fit={np.mean(Lus):.3f} Bi_m_fit={np.mean(Bims):.3f} | "
              f"relL2 calib={np.mean(rel):.4f} nominal={relL2_nom:.4f}")

    passing = [r["pn"] for r in rows if r["frac_pass"] >= 0.5]
    pn_silent = max(passing) if passing else 0.0
    print(f"\n=== silent band: the calibrated reduced model passes chi2 GOF (alpha={ALPHA}, "
          f">=50% of seeds) up to Pn = {pn_silent:.4f} ===")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(theta_true=THETA_TRUE, K=K_DEFAULT, noise_frac=FRAC,
                                   seeds=len(SEEDS), alpha=ALPHA, dof=2 * K_DEFAULT - 2,
                                   pn_silent=pn_silent, rows=rows), indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
