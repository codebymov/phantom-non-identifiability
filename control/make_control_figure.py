"""Figure: the dropped coupling removes the actuator, and a controller built on it pays for that.

(a) Sensitivity of the moisture observables to the manipulated air temperature u, versus the Posnov
    coupling. Exactly zero at Pn=0 (the reduction) and O(Pn) thereafter -- the same first-order law
    the parameter-to-observation Jacobian obeys. The surface-temperature sensitivity is flat and
    large throughout: the actuator never loses authority over temperature, only over drying.
(b) Closed-loop cost on a coupled truth plant for controllers differing only in their prediction
    model's Pn. The reduced-model controller's regret grows with the true coupling, while the
    control-sensitive temperature term under the reduced model stays excellent -- the control-side
    analogue of the paper's silent failure.

Run:  PYTHONPATH=. .venv/bin/python control/make_control_figure.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ij = json.loads((ROOT / "runs/control/input_jacobian.json").read_text())
sm = json.loads((ROOT / "runs/control/structural_mpc.json").read_text())

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 3.1))

# ---- (a) input sensitivity vs Pn ----
pn = np.array([r["pn"] for r in ij["rows"]])
dM = np.array([r["dM"] for r in ij["rows"]])
dT = np.array([r["dT"] for r in ij["rows"]])
ax.loglog(pn, dM, "o-", ms=4, color="#C0504D", label=r"$\|\partial \bar M/\partial u\|$ (drying)")
ax.loglog(pn, dT, "s-", ms=4, color="#1F77B4", label=r"$\|\partial T(0,\tau)/\partial u\|$ (temperature)")
ax.loglog(pn, dM[0] * (pn / pn[0]), "--", lw=1, color="0.45", label=r"slope $+1$ (first order)")
ax.axvspan(0.2, 2.0, color="#4C9F70", alpha=0.13, label="assumed paper range")
ax.annotate("at Pn = 0 the drying\nsensitivity is exactly 0",
            xy=(1.2e-3, 3e-3), xytext=(1.5e-3, 3e-1), fontsize=7.5, color="#7a2520",
            arrowprops=dict(arrowstyle="->", color="#7a2520", lw=0.9))
ax.set_xlabel("Posnov number  Pn"); ax.set_ylabel(r"$\|\partial(\cdot)/\partial u\|$")
ax.set_title("(a) the reduction deletes the actuator's drying authority", fontsize=10)
ax.legend(fontsize=7, loc="lower right", framealpha=0.9); ax.grid(True, which="both", alpha=0.25)

# ---- (b) closed-loop cost and regret ----
pns = sorted(float(k) for k in sm["results"])
get = lambda kind, key: np.array([sm["results"][f"{p:g}"][kind]["metrics"][key] for p in pns])
bx.semilogy(pns, get("reduced", "J_true"), "o-", ms=4, color="#C0504D",
            label="reduced prediction model (Pn=0)")
bx.semilogy(pns, get("matched", "J_true"), "s-", ms=4, color="#2E7D32",
            label="matched prediction model")
bx.semilogy(pns, get("uncertain", "J_true"), "^--", ms=4, color="#1F77B4",
            label="model-uncertain (spike-and-slab)")
bx.semilogy(pns, get("reduced", "temp_track"), ":", lw=1.6, color="#8a6d3b",
            label="control-sensitive temperature term\n(reduced model)")
bx.axvspan(0.10, 0.35, color="#C0504D", alpha=0.12, label="silent band (Fig. 1b)")
# Honest marker: with a fixed moisture target the reachable set shifts with the coupling, so at the
# extremes the target lies outside it and the comparison degenerates. Those points measure
# reachability, not model quality, and are excluded from the regret claim.
unreach = [p for p in pns if sm["results"][f"{p:g}"]["matched"]["metrics"]["M_err"] > 0.02]
if unreach:
    bx.plot(unreach, [sm["results"][f"{p:g}"]["matched"]["metrics"]["J_true"] for p in unreach],
            "x", ms=11, mew=2.0, color="0.35", zorder=5,
            label="target outside reachable set")

bx.set_xlabel("true Posnov number  Pn  of the plant")
bx.set_ylabel("closed-loop cost on the truth plant")
bx.set_title("(b) the controller pays, on a term its model treats as insensitive", fontsize=10)
bx.legend(fontsize=6.8, loc="upper left", framealpha=0.9); bx.grid(True, which="both", alpha=0.25)

fig.tight_layout()
out = ROOT / "paper/ml4ps/figures/C_control_structure.png"
fig.savefig(out, dpi=200)
print("saved", out)
print("\nreachability check (matched-controller terminal moisture error):")
for p in pns:
    e = sm["results"][f"{p:g}"]["matched"]["metrics"]["M_err"]
    print(f"  Pn={p:.2f}  |M err|={e:.4f}" + ("   <- target NOT reachable" if e > 0.02 else ""))
print()
for p in pns:
    r = sm["results"][f"{p:g}"]
    print(f"  Pn={p:.2f}  regret {r['reduced']['metrics']['J_true']/r['matched']['metrics']['J_true']:.1f}x  "
          f"u_mean reduced {r['reduced']['metrics']['u_mean']:.3f} vs matched "
          f"{r['matched']['metrics']['u_mean']:.3f}")
