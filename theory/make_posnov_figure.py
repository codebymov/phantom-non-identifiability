"""Figure: the phantom boundary dissolves with the Posnov coupling, and is silent while it lasts.

(a) Moisture-only Cramer-Rao relative std for {Ko, Bi_t} vs. the Posnov coupling, with the assumed
    physical range and the crossover distribution over the operating box shaded. At Pn=0 the
    parameters are unidentifiable (structural null); for any Pn>0 the rank is full.
(b) The failure is SILENT. The reduced (Pn=0) model is calibrated over the parameters it can see
    (Lu, Bi_m) against the same noisy moisture data and tested for goodness of fit. Over a wide band
    of true couplings it passes -- statistically indistinguishable from a correct model -- while
    still reporting rank 2, CR = infinity and "add a temperature channel".

Uses the shared observation model (theory/observe.py).
Run:  PYTHONPATH=. .venv/bin/python theory/make_posnov_figure.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from theory.observe import cr

ROOT = Path(__file__).resolve().parents[1]
TH0 = (0.5, 2.0, 1.0, 1.0)

pns = np.concatenate([[0.0], np.geomspace(0.005, 2.5, 44)])
pos = pns > 0
crKo = np.array([cr(TH0, p, "Ko") for p in pns[pos]])
crBit = np.array([cr(TH0, p, "Bi_t") for p in pns[pos]])

wc = json.loads((ROOT / "runs/theory/weak_coupling.json").read_text())
q = wc["sobol"]["quantiles"]
bmin, bmax = q["p5"], q["p95"]
sr = json.loads((ROOT / "runs/theory/silent_regime.json").read_text())
rows = [r for r in sr["rows"] if r["pn"] > 0]
p_pn = np.array([r["pn"] for r in rows]); p_val = np.array([r["p"] for r in rows])
pn_silent = sr["pn_silent"]
pn_star = float(pns[pos][np.argmax(crKo < 100)])

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 2.95))

ax.axvspan(0.2, 2.0, color="#4C9F70", alpha=0.13, label="assumed paper range (Pn 0.2 to 2)")
ax.axvspan(bmin, bmax, color="#8888aa", alpha=0.18,
           label=f"crossover, p5-p95 over the box ({bmin:.3f} to {bmax:.2f})")
ax.axhline(100, color="0.4", ls="--", lw=1)
ax.text(0.006, 122, "100% CR (unusable above)", fontsize=8, color="0.35")
ax.loglog(pns[pos], crKo, "o-", ms=4, color="#C0504D", label=r"$\mathrm{Ko}$ (moisture-only CR)")
ax.loglog(pns[pos], crBit, "s-", ms=4, color="#1F77B4", label=r"$\mathrm{Bi}_t$ (moisture-only CR)")
ax.text(0.0055, 2.0, "Pn=0: rank 2,\n{Ko,Bi$_t$} unobservable", fontsize=8, color="#5c8a2a")
ax.set_xlabel("Posnov number  Pn  (thermogradient coupling strength)")
ax.set_ylabel("moisture-only Cramér-Rao rel. std [%]")
ax.set_title("(a) the boundary dissolves for any Pn > 0", fontsize=10)
ax.set_ylim(0.5, 1e4); ax.set_xlim(4e-3, 2.6)
ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
ax.grid(True, which="both", alpha=0.25)

bx.axvspan(pn_star, pn_silent, color="#C0504D", alpha=0.16,
           label=f"silent-failure window ({pn_star:.2f} to {pn_silent:.2f})")
bx.axvspan(0.2, 2.0, color="#4C9F70", alpha=0.13, label="assumed paper range")
bx.semilogx(p_pn, np.clip(p_val, 1e-6, None), "o-", ms=4, color="#333333",
            label="reduced model, $\\chi^2$ goodness of fit")
bx.axhline(sr["alpha"], color="#C0504D", ls="--", lw=1.2)
bx.text(6e-3, 0.075, "reject at p = 0.05", fontsize=8, color="#C0504D")
bx.text(5.5e-3, 2.5e-3, "inside the window the reduced model\nfits the data, yet still reports rank 2\n"
        "and prescribes a temperature sensor", fontsize=7.5, color="#7a2520")
bx.set_yscale("log"); bx.set_ylim(1e-6, 3); bx.set_xlim(4e-3, 2.6)
bx.set_xlabel("true Posnov number  Pn  (of the data-generating system)")
bx.set_ylabel("$p$-value (calibrated reduced model)")
bx.set_title("(b) inside the window the reduced model looks correct", fontsize=10)
bx.legend(fontsize=7, loc="lower left", framealpha=0.9)
bx.grid(True, which="both", alpha=0.25)

fig.tight_layout()
out = ROOT / "paper/ml4ps/figures/P_posnov_crossover.png"
fig.savefig(out, dpi=200)
print("saved", out)
print(f"nominal crossover {pn_star:.3f}; silent to {pn_silent:.3f}; box p5-p95 {bmin:.4f}-{bmax:.4f}")
