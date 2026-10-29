"""Stage-1 Posnov check: does the moisture-only observability null survive Pn > 0?

The reduced-model result rests on the moisture balance being AUTONOMOUS at Pn = 0, which makes
dH_M/dKo = dH_M/dBi_t == 0 for any moisture-only sensor. The full Luikov system carries a
thermogradient (Soret) cross-term controlled by the Posnov number Pn (Mikhailov & Ozisik 1984;
Cotta 1993; Lugon & Silva Neto, COBEM 2003, eqs. 1a-1b). This script restates the identifiability
analysis as a function of Pn, under the shared observation model (theory/observe.py): mean moisture
+ surface mass flux, independent per-channel noise at 1% of each channel's range.

Two pre-registered gates are asserted below:
  * at Pn = 0 the {Ko, Bi_t} columns are exactly zero and rank(S) = 2 (the phantom null), which
    holds for the physically correct flux channel just as it did for the earlier rate channel;
  * for Pn > 0 the rank is full.

Run:  PYTHONPATH=. .venv/bin/python theory/posnov_identifiability_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from theory.observe import NAMES4, fisher

THETA0 = (0.5, 2.0, 1.0, 1.0)
PN_GRID = [0.0, 0.05, 0.1, 0.2, 0.6, 1.0, 2.0]
COL_ZERO_REL = 1e-6      # column "effectively zero" relative to the largest column
CR_DIVERGENT = 100.0     # CR relative std > 100% == unusable
OUT = Path(__file__).resolve().parents[1] / "runs/theory/fisher_nullspace.json"


def main() -> None:
    r0 = fisher(THETA0, 0.0)
    rel0 = r0["col_norm"] / r0["col_norm"].max()
    assert r0["rank"] == 2, r0["rank"]
    assert rel0[1] < COL_ZERO_REL and rel0[2] < COL_ZERO_REL, rel0
    print("gate OK @ Pn=0: rank(S)=2 and the {Ko, Bi_t} columns are exactly zero, with the\n"
          "physically correct surface-flux channel (the flux is -Lu*Bi_m*M(0) at Pn=0, autonomous).\n")

    print(f"MOISTURE-ONLY (mean moisture + surface mass flux), theta={dict(zip(NAMES4, THETA0))}\n")
    hdr = (f"{'Pn':>5} | {'||dH/dlogKo||':>14} {'||dH/dlogBit||':>15} | {'rank':>4} | "
           f"{'CR% Ko':>8} {'CR% Bi_t':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for pn in PN_GRID:
        r = fisher(THETA0, pn)
        cn, crb = r["col_norm"], r["crb"]
        f = lambda v, w: (f"{v:{w}.1f}" if np.isfinite(v) else f"{'inf':>{w}}")
        print(f"{pn:5.2f} | {cn[1]:14.4e} {cn[2]:15.4e} | {r['rank']:4d} | "
              f"{f(crb[1],8)} {f(crb[2],9)}")
        if pn > 0:
            assert r["rank"] == 4, (pn, r["rank"])
        rows.append(dict(pn=pn, rank=r["rank"], col_norm=cn.tolist(), crb=crb.tolist()))

    print("\nCONTROL (moisture + one mean-temperature channel):")
    for pn in PN_GRID:
        r = fisher(THETA0, pn, with_T=True)
        print(f"  Pn={pn:4.2f}  rank={r['rank']}  " +
              "  ".join(f"{n}={c:6.1f}%" for n, c in zip(NAMES4, r["crb"])))

    print("\n=== outcome classification (pre-registered thresholds) ===")
    for row in rows:
        if row["pn"] < 0.2:
            continue
        ko, bit = row["crb"][1], row["crb"][2]
        res = np.isfinite(ko) and ko < CR_DIVERGENT and np.isfinite(bit) and bit < CR_DIVERGENT
        oc = ("C (Ko & Bi_t resolvable -> the null is a model-reduction artefact)" if res
              else "B (full rank but CR still >100%: practical, not structural)")
        print(f"  Pn={row['pn']:4.2f}: {oc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(theta0=dict(zip(NAMES4, THETA0)), rows=rows), indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
