"""Figure for the Luikov identifiability result (point 2)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "runs/theory/identifiability.npz")
OUT = ROOT / "report/figures/K_identifiability.png"
names = [str(x) for x in d["names"]]
ACCENT, WARM = "#1f6fb2", "#e08020"


def main() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.0))

    x = np.arange(4); w = 0.38
    axA.bar(x - w/2, d["moist_cols"], w, color=ACCENT, label="moisture cell (scale + mass-flow)")
    axA.bar(x + w/2, d["temp_cols"], w, color=WARM, label="+ temperature channel")
    axA.set_xticks(x); axA.set_xticklabels(names)
    axA.set_ylabel(r"sensitivity  $\|\partial g/\partial\log\theta_j\|$")
    axA.set_title("Structural identifiability: column norms\n"
                  "(moisture sensors are blind to Ko, Bi$_t$)", fontsize=10)
    axA.legend(fontsize=8.5, loc="upper center")
    for k in (1, 2):  # Ko, Bi_t
        axA.annotate("0 (unobservable)", (k - w/2, 0), (k - w/2, d["temp_cols"].max()*0.25),
                     fontsize=8, color=ACCENT, ha="center", rotation=90,
                     arrowprops=dict(arrowstyle="->", color=ACCENT))

    # Identifiability spectrum (eigenvalues); zeros placed at a floor for the log axis.
    floor = 1e-1
    me = np.where(d["moist_eig"] > 1e-6 * d["moist_eig"].max(), d["moist_eig"], floor)
    te = np.where(d["temp_eig"] > 1e-6 * d["temp_eig"].max(), d["temp_eig"], floor)
    idx = np.arange(4)
    axB.semilogy(idx, me, "o-", color=ACCENT, ms=7, label=f"moisture cell (rank {int(d['moist_rank'])})")
    axB.semilogy(idx, te, "s--", color=WARM, ms=6, label=f"+ temperature (rank {int(d['temp_rank'])})")
    axB.axhline(floor, color="grey", ls=":", lw=1)
    axB.text(2.0, floor*1.3, "null directions {Ko, Bi$_t$}", fontsize=8.5, color=ACCENT)
    axB.set_xticks(idx); axB.set_xticklabels([f"$\\mu_{i+1}$" for i in idx])
    axB.set_ylabel("FIM eigenvalue (identifiability)")
    axB.set_title("Identifiability spectrum of the Fisher matrix\n"
                  "two zero modes collapse without a T sensor", fontsize=10)
    axB.legend(fontsize=8.5)

    fig.suptitle("Why the drying cell identifies moisture but not temperature parameters",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
