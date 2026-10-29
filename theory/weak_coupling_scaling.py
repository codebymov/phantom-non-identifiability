"""Weak-coupling scaling law, one-probe crossover prediction, and the box crossover distribution.

Expand the moisture field to first order in the Posnov number about the decoupled solution,

    M = M_0 + Pn * M_1 + O(Pn^2),
    d_tau M_1 = Lu d_xixi M_1 + Lu d_xixi T_0,     M_1(.,0) = 0,

where T_0 is the Pn=0 temperature field and therefore carries Ko and Bi_t. Both moisture channels
inherit the parameter dependence only through the Pn * M_1 term (the surface flux additionally
through its explicit Pn Bi_t dT/dxi term, which is also O(Pn)), so

    dH_M / dlog theta_j = Pn * a_j(theta) + O(Pn^2),
    F_jj                = Pn^2 a_j^2 / sigma^2,
    CR(theta_j)         = gamma_j(theta) / Pn + O(1).

Tested here: (1) the O(Pn) / O(1/Pn) exponents; (2) that a single probe solve at small Pn predicts
the crossover Pn* = gamma_j, turning the sweep diagnostic from exploratory into predictive; (3) the
distribution of Pn* over the operating box, by Sobol sampling rather than corners alone.

Uses the shared observation model (theory/observe.py): mean moisture + surface mass flux, with
independent per-channel noise at 1% of each channel's range.

Run:  .venv/bin/python theory/weak_coupling_scaling.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from theory.observe import cr

BOX = {"Lu": (0.2, 1.0), "Ko": (1.0, 3.0), "Bi_t": (0.5, 2.0), "Bi_m": (0.5, 2.0)}
ORDER = ["Lu", "Ko", "Bi_t", "Bi_m"]
PN_PROBE = 0.02
N_SOBOL = 256
OUT = Path(__file__).resolve().parents[1] / "runs/theory/weak_coupling.json"


def crossover(theta, which="Ko", pn_max=2.5, n=40):
    """First Pn on a log grid where CR drops below 100%, refined by log-log interpolation."""
    prev_p = prev_c = None
    for p in np.geomspace(1e-4, pn_max, n):
        c = cr(theta, p, which)
        if not np.isfinite(c):
            prev_p = prev_c = None
            continue
        if c < 100.0:
            if prev_p is None:
                return float(p)
            t = (np.log(100.0) - np.log(prev_c)) / (np.log(c) - np.log(prev_c))
            return float(np.exp(np.log(prev_p) + t * (np.log(p) - np.log(prev_p))))
        prev_p, prev_c = p, c
    return np.nan


def main() -> None:
    th0 = (0.5, 2.0, 1.0, 1.0)

    # ---- (1) weak-coupling exponents ----
    pns = np.geomspace(1e-3, 0.1, 9)
    rKo = np.array([cr(th0, p, "Ko") for p in pns])
    rBit = np.array([cr(th0, p, "Bi_t") for p in pns])
    lp = np.log(pns)
    sKo = float(np.polyfit(lp, np.log(rKo), 1)[0])
    sBit = float(np.polyfit(lp, np.log(rBit), 1)[0])
    print("=== weak-coupling exponents at nominal theta (Pn in [1e-3, 0.1]) ===")
    print(f"  d log CR(Ko)   / d log Pn = {sKo:+.3f}   (theory -1)")
    print(f"  d log CR(Bi_t) / d log Pn = {sBit:+.3f}   (theory -1)")

    # ---- (2) one-probe prediction, validated at the 16 corners ----
    print(f"\n=== one-probe crossover prediction (probe Pn = {PN_PROBE}) ===")
    print(f"  {'Lu':>5}{'Ko':>5}{'Bi_t':>6}{'Bi_m':>6} | {'Pn*_pred':>9} {'Pn*_meas':>9} {'ratio':>7}")
    corners, preds, meas = [], [], []
    for combo in itertools.product(*[BOX[k] for k in ORDER]):
        pred = cr(combo, PN_PROBE, "Ko") * PN_PROBE / 100.0   # CR in %, threshold 100%
        m = crossover(combo)
        corners.append(dict(zip(ORDER, combo)) | dict(pred=float(pred), meas=m))
        preds.append(pred); meas.append(m)
        r = pred / m if np.isfinite(m) else np.nan
        print(f"  {combo[0]:5.2f}{combo[1]:5.1f}{combo[2]:6.2f}{combo[3]:6.2f} | "
              f"{pred:9.4f} {m:9.4f} {r:7.3f}")
    preds, meas = np.array(preds), np.array(meas)
    ok = np.isfinite(meas)
    err = np.abs(preds[ok] / meas[ok] - 1) * 100
    small = preds[ok] < 0.05
    print(f"\n  prediction error over {ok.sum()} corners: median {np.median(err):.1f}%, "
          f"max {err.max():.1f}%")
    if small.any() and (~small).any():
        print(f"    Pn*_pred < 0.05 : median {np.median(err[small]):.1f}%, max {err[small].max():.1f}%")
        print(f"    Pn*_pred >= 0.05: median {np.median(err[~small]):.1f}%, max {err[~small].max():.1f}%"
              "   (first-order expansion no longer holds at an O(1) crossover)")

    # ---- (3) Sobol distribution of the crossover over the box ----
    print(f"\n=== crossover distribution over the operating box ({N_SOBOL} Sobol points) ===")
    lo = np.array([BOX[k][0] for k in ORDER]); hi = np.array([BOX[k][1] for k in ORDER])
    pts = qmc.scale(qmc.Sobol(4, scramble=True, seed=0).random(N_SOBOL), lo, hi)
    xs = np.array([crossover(tuple(p)) for p in pts])
    good = np.isfinite(xs)
    q = np.percentile(xs[good], [5, 25, 50, 75, 95])
    print(f"  n = {good.sum()}/{N_SOBOL} with a crossover below Pn = 2.5")
    print(f"  min {xs[good].min():.4f} | p5 {q[0]:.4f} | p25 {q[1]:.4f} | median {q[2]:.4f} | "
          f"p75 {q[3]:.4f} | p95 {q[4]:.4f} | max {xs[good].max():.4f}")
    print(f"  fraction with Pn* below the assumed physical range (0.2): "
          f"{float((xs[good] < 0.2).mean())*100:.0f}%")

    # ---- (4) compact power law ----
    X = np.column_stack([np.ones(good.sum())] + [np.log(pts[good][:, i]) for i in range(4)])
    coef, *_ = np.linalg.lstsq(X, np.log(xs[good]), rcond=None)
    rel = np.abs(np.exp(X @ coef) / xs[good] - 1) * 100
    print("\n=== compact power law over the box ===")
    print(f"  Pn* ~ {np.exp(coef[0]):.4f} * " +
          " ".join(f"{k}^({coef[i+1]:+.2f})" for i, k in enumerate(ORDER)))
    print(f"  fit error: median {np.median(rel):.1f}%, max {rel.max():.1f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        theta0=th0, pn_probe=PN_PROBE, slopes=dict(cr_Ko=sKo, cr_Bit=sBit), corners=corners,
        pred_err_pct=dict(median=float(np.median(err)), max=float(err.max()), n=int(ok.sum())),
        sobol=dict(n=int(good.sum()), quantiles=dict(zip(["p5","p25","p50","p75","p95"], q.tolist())),
                   min=float(xs[good].min()), max=float(xs[good].max()),
                   frac_below_0p2=float((xs[good] < 0.2).mean())),
        power_law=dict(coef=coef.tolist(), terms=["const"] + ORDER,
                       fit_err_pct=dict(median=float(np.median(rel)), max=float(rel.max())))),
        indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
