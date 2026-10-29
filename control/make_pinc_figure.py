"""Figure for the PINC-NMPC on the real Luikov system (constraint satisfaction vs PI)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "pinc_result.npz")
OUT = HERE.parent / "report/figures/P_pinc_nmpc.png"
DT = float(d["dtau"]); TSET, TCRIT = 0.70, float(d["t_crit"])
PI, MPC = "#c0392b", "#2ca02c"
t = np.arange(len(d["PID_u"])) * DT


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.axhspan(TCRIT, 0.75, color="red", alpha=0.08)
    ax1.axhline(TCRIT, color="red", ls="--", lw=1.3, label=f"$T_{{crit}}$ = {TCRIT} (Beschränkung)")
    ax1.axhline(TSET, color="k", ls=":", lw=1.4, label=f"Sollwert $w$ = {TSET} (above the limit)")
    ax1.plot(t, d["PID_Tsurf"], color=PI, lw=2, label="PI: surface Istwert (scorches, +0.09 over limit)")
    ax1.plot(t, d["PINC-NMPC_Tsurf"], color=MPC, lw=2, label="PINC-NMPC: respects $T_{crit}$")
    ax1.set_ylabel("surface temperature $T_0$ (dimensionless)")
    ax1.set_ylim(0, 0.73)
    ax1.set_title("PINC-NMPC on the real Luikov system: enforces the scorch limit a PI controller violates\n"
                  "(control-oriented surrogate trained on the physics solver; real-time, differentiable)",
                  fontsize=10.3)
    ax1.legend(fontsize=8.2, loc="lower right"); ax1.grid(alpha=0.25)

    ax2.plot(t, d["PID_u"], color=PI, lw=1.8, label="PI")
    ax2.plot(t, d["PINC-NMPC_u"], color=MPC, lw=1.8, label="PINC-NMPC")
    ax2.set_ylabel("hot-air temp $u$\n(Stellgröße)")
    ax2.set_xlabel(r"dimensionless time $\tau$")
    ax2.legend(fontsize=8.5, loc="upper right"); ax2.grid(alpha=0.25)
    # note: mean moisture Mbar is identical for both controllers (autonomous in the simplified system)
    dm = float(np.max(np.abs(d["PID_Mbar"] - d["PINC-NMPC_Mbar"])))
    ax2.text(0.99, -0.42, f"(drying $\\bar M$ identical for both, $\\Delta<{max(dm,1e-3):.0e}$: "
             "moisture is autonomous -- air temp controls $T$, not drying rate)",
             transform=ax2.transAxes, ha="right", fontsize=7.6, color="grey")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
