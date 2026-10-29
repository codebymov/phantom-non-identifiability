"""M0b -- the metrics table, recomputed the way anyone would actually validate a model.

The first pass compared the reduced simulator against the parent at the *same* parameters. That
overstates the forward error, because a practitioner calibrates first, and (as Sec. 5 of the
predecessor paper shows) calibration launders most of the discrepancy into the parameters the reduced
model retains. The comparison that characterises the failure is therefore:

  fit (Lu, Bi_m) under the reduced model to the parent's noisy observations, then report
  (a) the POST-CALIBRATION forward discrepancy, and
  (b) the tangent metrics of the *calibrated* reduced model against the parent.

Note what this is and is not: it measures model-form error of a reduction, not approximation error of
a surrogate. Those are different failures and the paper keeps them apart.

Run: PYTHONPATH=. .venv/bin/python tangentsafe/experiments/m0b_calibrated_metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tangentsafe.audit.metrics import tangent_metrics
from theory.observe import FRAC, K_DEFAULT, channels, sample_times, sensitivity, sigmas, solve

NAMES = ["Lu", "Ko", "Bi_t", "Bi_m"]
TH = dict(Lu=0.5, Ko=2.0, Bi_t=1.0, Bi_m=1.0)
PN_GRID = [0.05, 0.1, 0.2, 0.35, 0.6, 1.0]
OUT = Path(__file__).resolve().parents[2] / "runs/tangentsafe/m0b_calibrated.json"


def jac(theta, pn):
    S, sig = sensitivity([theta["Lu"], theta["Ko"], theta["Bi_t"], theta["Bi_m"]], pn)
    return S, sig


def main() -> None:
    print("Reduced model vs parent, BEFORE and AFTER calibrating the parameters it can see.\n")
    hdr = (f"{'Pn'::>0}" if False else
           f"{'Pn':>5} {'fwd raw':>8} {'fwd calib':>10} {'Lu fit':>7} {'Bi_m fit':>9} "
           f"{'angle deg':>10} {'E_J':>7} {'rank':>6} {'order?':>7}")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for pn in PN_GRID:
        par = solve([TH["Lu"], TH["Ko"], TH["Bi_t"], TH["Bi_m"]], pn)
        ts = sample_times(par, K_DEFAULT)
        y = channels(par, ts)
        sig = sigmas(y)
        y_obs = np.concatenate(y)

        def resid(logp):
            Lu, Bim = np.exp(logp)
            pred = channels(solve([Lu, TH["Ko"], TH["Bi_t"], Bim], 0.0), ts)
            return np.concatenate([(p - o) / s for p, o, s in zip(pred, y, sig)])

        r = least_squares(resid, np.log([TH["Lu"], TH["Bi_m"]]), method="trf",
                          bounds=(np.log([0.05, 0.1]), np.log([2.0, 10.0])), max_nfev=300)
        Lu_f, Bim_f = np.exp(r.x)
        th_cal = dict(TH, Lu=Lu_f, Bi_m=Bim_f)

        raw = np.concatenate(channels(solve([TH["Lu"], TH["Ko"], TH["Bi_t"], TH["Bi_m"]], 0.0), ts))
        cal = np.concatenate(channels(solve([Lu_f, TH["Ko"], TH["Bi_t"], Bim_f], 0.0), ts))
        f_raw = float(np.linalg.norm(raw - y_obs) / np.linalg.norm(y_obs))
        f_cal = float(np.linalg.norm(cal - y_obs) / np.linalg.norm(y_obs))

        J_par, sg = jac(TH, pn)
        J_red, _ = jac(th_cal, 0.0)          # calibrated reduced model's tangent map
        m = tangent_metrics(J_red, J_par, sg, NAMES, k=2, fwd_relL2=f_cal)
        m["fwd_raw"] = f_raw; m["Lu_fit"] = float(Lu_f); m["Bim_fit"] = float(Bim_f)
        rows[str(pn)] = m
        print(f"{pn:5.2f} {f_raw:8.4f} {f_cal:10.4f} {Lu_f:7.3f} {Bim_f:9.3f} "
              f"{m['subspace_angle_deg']:10.2f} {m['E_J']:7.3f} "
              f"{m['rank_surrogate']:3d}/{m['rank_reference']:<2d} {m['order_agrees']!s:>7}")

    print("\n  Calibration removes most of the forward discrepancy while the tangent map stays")
    print("  wrong: that pairing, not the raw error, is what a validation exercise would see.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(theta=TH, noise_frac=FRAC, K=K_DEFAULT, rows=rows), indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
