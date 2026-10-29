"""Does the dropped Posnov coupling also delete the actuator's authority over drying?

The paper shows that setting Pn = 0 zeros the parameter-to-observation Jacobian columns for
{Ko, Bi_t}. The same reduction makes the moisture balance autonomous, so it must equally zero the
INPUT-to-observation Jacobian: with u the manipulated hot-air temperature and H_rate the drying-rate
observation,

    dH_rate/du |_{Pn=0} = 0    exactly,

because M is then advanced by an operator that references neither u nor T (see
control/luikov_plant.py step()). This script (a) verifies that zero numerically, and (b) tests --
rather than assumes -- the weak-coupling hypothesis ||dH/du|| = O(Pn), by finite differences through
the plant over a sweep of coupling strengths.

Observables: mean moisture Mbar(tau), drying rate dMbar/dtau (= the surface mass flux), and surface
temperature Tsurf(tau) as the control-relevant safety output.

Run:  PYTHONPATH=. .venv/bin/python control/input_jacobian_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from control.coupled_plant import make_plant

DTAU, NSTEP = 0.01, 200
U0, DU = 1.0, 1e-4
OUT = Path(__file__).resolve().parents[1] / "runs/control/input_jacobian.json"


def rollout(pn: float, u: float):
    """Constant-u open-loop rollout; returns Mbar(tau), drying rate, Tsurf(tau)."""
    pl = make_plant(pn, dtau=DTAU); pl.reset()
    Mbar, Tsurf = [], []
    for _ in range(NSTEP):
        s = pl.step(u)
        Mbar.append(s[0]); Tsurf.append(s[4])
    Mbar = np.array(Mbar); Tsurf = np.array(Tsurf)
    rate = np.gradient(Mbar, DTAU)
    return Mbar, rate, Tsurf


def dH_du(pn: float):
    """Central-difference sensitivity of each observable trajectory to the constant control u."""
    Mp, Rp, Tp = rollout(pn, U0 + DU)
    Mm, Rm, Tm = rollout(pn, U0 - DU)
    return ((Mp - Mm) / (2 * DU), (Rp - Rm) / (2 * DU), (Tp - Tm) / (2 * DU))


def main() -> None:
    print("Sensitivity of the drying observables to the manipulated air temperature u.\n")
    dM0, dR0, dT0 = dH_du(0.0)
    print("Pn = 0 (the reduction):")
    print(f"  ||d Mbar / du||      = {np.linalg.norm(dM0):.3e}   (exact zero expected)")
    print(f"  ||d rate / du||      = {np.linalg.norm(dR0):.3e}   (exact zero expected)")
    print(f"  ||d Tsurf / du||     = {np.linalg.norm(dT0):.3e}   (nonzero: u does drive temperature)")
    assert np.all(dM0 == 0.0) and np.all(dR0 == 0.0), "moisture sensitivity is not exactly zero"
    assert np.linalg.norm(dT0) > 1.0, "temperature sensitivity vanished -- actuator is dead"
    print("  -> the actuator retains full authority over temperature and exactly none over moisture.\n")

    pns = np.geomspace(1e-3, 1.0, 13)
    rows = []
    print(f"{'Pn':>8} {'||dMbar/du||':>14} {'||drate/du||':>14} {'||dTsurf/du||':>14}")
    print("-" * 54)
    for pn in pns:
        dM, dR, dT = dH_du(pn)
        rows.append(dict(pn=float(pn), dM=float(np.linalg.norm(dM)), dR=float(np.linalg.norm(dR)),
                         dT=float(np.linalg.norm(dT))))
        print(f"{pn:8.4f} {np.linalg.norm(dM):14.5e} {np.linalg.norm(dR):14.5e} "
              f"{np.linalg.norm(dT):14.5e}")

    lp = np.log(pns)
    sM = float(np.polyfit(lp, np.log([r["dM"] for r in rows]), 1)[0])
    sR = float(np.polyfit(lp, np.log([r["dR"] for r in rows]), 1)[0])
    # restrict to the genuinely weak-coupling decade for the first-order test
    m = pns <= 1e-1
    sM_w = float(np.polyfit(lp[m], np.log([r["dM"] for r, k in zip(rows, m) if k]), 1)[0])
    sR_w = float(np.polyfit(lp[m], np.log([r["dR"] for r, k in zip(rows, m) if k]), 1)[0])
    print(f"\nlog-log slope over the full sweep : dMbar {sM:+.3f}   drate {sR:+.3f}")
    print(f"log-log slope for Pn <= 0.1       : dMbar {sM_w:+.3f}   drate {sR_w:+.3f}   (first order -> +1)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(u0=U0, dtau=DTAU, nstep=NSTEP,
                                   zero_check=dict(dM=float(np.linalg.norm(dM0)),
                                                   dR=float(np.linalg.norm(dR0)),
                                                   dT=float(np.linalg.norm(dT0))),
                                   rows=rows,
                                   slopes=dict(dM_full=sM, dR_full=sR, dM_weak=sM_w, dR_weak=sR_w)),
                              indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
