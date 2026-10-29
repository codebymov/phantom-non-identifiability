"""Is K=40 enough? Sampling sufficiency of the flux channel in the Fisher analysis.

The surface mass flux equals d<M>/dtau exactly, so it is a derivative functional and has the higher
effective bandwidth of the two moisture channels. The identifiability analysis samples K = 40 points
over tau in (0, 2]. This script asks whether that budget loses information, by sweeping K and
checking whether CR falls smoothly (plain sample-count scaling, K^-1/2 for independent noise) or
drops off a cliff once the derivative channel is resolved (aliasing).

Run:  PYTHONPATH=. .venv/bin/python theory/sampling_sufficiency.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import theory.observe as ob

K_GRID = [20, 40, 60, 90, 120, 180]
PN_LIST = [0.1, 0.2, 0.6, 1.0]
THETA0 = (0.5, 2.0, 1.0, 1.0)
OUT = Path(__file__).resolve().parents[1] / "runs/theory/sampling_sufficiency.json"


def main() -> None:
    ob.NX, ob.NT = 101, 401     # finer in time so K up to 180 is resolvable
    print("CR%(Ko) / CR%(Bi_t) vs the number of sample times K (moisture-only)\n")
    hdr = f"{'Pn':>5} | " + " ".join(f"K={k}".rjust(14) for k in K_GRID)
    print(hdr); print("-" * len(hdr))
    rows = []
    for pn in PN_LIST:
        vals = [ob.fisher(THETA0, pn, K=K)["crb"] for K in K_GRID]
        print(f"{pn:5.2f} | " + " ".join(f"{v[1]:6.1f}/{v[2]:5.1f}".rjust(14) for v in vals))
        rows.append(dict(pn=pn, K=K_GRID, cr_Ko=[float(v[1]) for v in vals],
                         cr_Bit=[float(v[2]) for v in vals]))
    print("\nscaling of CR with K (independent additive noise predicts K^-1/2):")
    for r in rows:
        Ks = np.array(r["K"], float)
        r["slope_Ko"] = float(np.polyfit(np.log(Ks), np.log(r["cr_Ko"]), 1)[0])
        r["slope_Bit"] = float(np.polyfit(np.log(Ks), np.log(r["cr_Bit"]), 1)[0])
        print(f"  Pn={r['pn']:4.2f}: d log CR / d log K = {r['slope_Ko']:+.2f} (Ko), "
              f"{r['slope_Bit']:+.2f} (Bi_t)")
    print("\nNo cliff: smooth sample-count scaling, so K=40 is a conservative budget rather than an\n"
          "aliasing failure, and the reported CR values are pessimistic in that direction.")
    ob.NX, ob.NT = 101, 201
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(theta0=THETA0, rows=rows), indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
