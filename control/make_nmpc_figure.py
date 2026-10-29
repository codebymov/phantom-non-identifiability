"""Figure for the surrogate-NMPC proof-of-concept (constraint satisfaction vs PID)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "nmpc_result.npz")
OUT = HERE.parent / "report/figures/O_nmpc.png"
DT, TCRIT = 10.0, 72.0
t = d["tgrid"] * DT / 60.0     # minutes
PID, MPC = "#c0392b", "#2ca02c"


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.axhspan(TCRIT, 80, color="red", alpha=0.08)
    ax1.axhline(TCRIT, color="red", ls="--", lw=1.3, label=f"$T_{{crit}}$ = {TCRIT} °C (Beschränkung)")
    ax1.plot(t, d["PID_w"], "k:", lw=1.6, label="Sollwert $w$ (75 °C, above limit)")
    ax1.plot(t, d["PID_T"], color=PID, lw=2, label="PID: Istwert (scorches, +3 °C over limit)")
    ax1.plot(t, d["surrogate-NMPC_T"], color=MPC, lw=2, label="surrogate-NMPC: respects $T_{crit}$")
    ax1.set_ylabel("material temperature $T_G$ (°C)")
    ax1.set_ylim(40, 78)
    ax1.set_title("Surrogate-NMPC enforces the safety constraint a PID cannot\n"
                  "(aggressive setpoint above $T_{crit}$; real-time via a differentiable surrogate)",
                  fontsize=10.5)
    ax1.legend(fontsize=8.3, loc="lower right"); ax1.grid(alpha=0.25)

    ax2.plot(t, d["PID_u"], color=PID, lw=1.8, label="PID")
    ax2.plot(t, d["surrogate-NMPC_u"], color=MPC, lw=1.8, label="surrogate-NMPC")
    ax2.set_ylabel("heating $Q_{heat}$ (W)\n(Stellgröße $u$)")
    ax2.set_xlabel("time (min)")
    ax2.legend(fontsize=8.5, loc="upper right"); ax2.grid(alpha=0.25)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
