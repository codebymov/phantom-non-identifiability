"""How much is the temperature channel actually worth, as a function of Pn?

Under the reduced (Pn=0) simulator the answer is categorical: the moisture-only Fisher matrix is
rank 2, lambda_min = 0, and one temperature channel is the difference between "impossible" and
"possible", so the E-optimal design step rates it mandatory. This script replaces that categorical
verdict with a graded one -- the RATIO of moisture-only to moisture-plus-temperature Cramer-Rao
bounds as the thermogradient coupling is restored. Uses the shared observation model.

Run:  PYTHONPATH=. .venv/bin/python theory/temperature_channel_value.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from theory.observe import fisher

THETA0 = (0.5, 2.0, 1.0, 1.0)
PN_GRID = [0.0, 0.05, 0.1, 0.2, 0.6, 1.0, 2.0]
OUT = Path(__file__).resolve().parents[1] / "runs/theory/temperature_channel_value.json"


def main() -> None:
    print("theta=(0.5,2,1,1), K=40, per-channel noise 1% of range. "
          "Gain = moisture-only / (moisture + mean T).\n")
    hdr = (f"{'Pn':>5} | {'lam_min(M)':>11} {'lam_min(M+T)':>13} | {'CR Ko M':>8} {'M+T':>7} "
           f"{'gain':>6} | {'CR Bit M':>9} {'M+T':>7} {'gain':>6}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for pn in PN_GRID:
        m, t = fisher(THETA0, pn), fisher(THETA0, pn, with_T=True)
        lm, lt = float(max(m["eig"].min(), 0.0)), float(max(t["eig"].min(), 0.0))
        gKo = m["crb"][1] / t["crb"][1] if np.isfinite(m["crb"][1]) else np.inf
        gBit = m["crb"][2] / t["crb"][2] if np.isfinite(m["crb"][2]) else np.inf
        f = lambda v, w=7: (f"{v:{w}.1f}" if np.isfinite(v) else f"{'inf':>{w}}")
        print(f"{pn:5.2f} | {lm:11.3e} {lt:13.3e} | {f(m['crb'][1],8)} {f(t['crb'][1])} "
              f"{f(gKo,6)} | {f(m['crb'][2],9)} {f(t['crb'][2])} {f(gBit,6)}")
        rows.append(dict(pn=pn, lam_min_M=lm, lam_min_MT=lt,
                         cr_Ko_M=float(m["crb"][1]), cr_Ko_MT=float(t["crb"][1]),
                         cr_Bit_M=float(m["crb"][2]), cr_Bit_MT=float(t["crb"][2]),
                         gain_Ko=float(gKo), gain_Bit=float(gBit)))
    phys = [r for r in rows if r["pn"] >= 0.2 and np.isfinite(r["gain_Ko"])]
    print(f"\nAt Pn=0 the gain is infinite (structural). Over the assumed physical range "
          f"Pn in [0.2, 2] it is {min(r['gain_Ko'] for r in phys):.1f}x to "
          f"{max(r['gain_Ko'] for r in phys):.1f}x on Ko and "
          f"{min(r['gain_Bit'] for r in phys):.1f}x to "
          f"{max(r['gain_Bit'] for r in phys):.1f}x on Bi_t: a conditioning improver, not a "
          "structural necessity.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(theta0=THETA0, rows=rows), indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
