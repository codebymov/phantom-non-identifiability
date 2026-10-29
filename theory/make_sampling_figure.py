"""Figure for the Shannon-sampling / Sobolev result (point 3)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "runs/theory/sampling.npz")
OUT = ROOT / "report/figures/L_sampling.png"
CM, CR = "#1f6fb2", "#c0392b"
DTAU_PAPER = 2.0 / 40   # the paper's K=41 measurement sampling over tau in [0,2]


def main() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.0))

    # Spectra (normalized), with effective bandwidths.
    PM = d["PM"] / d["PM"].max(); PR = d["PR"] / d["PR"].max()
    axA.semilogy(d["fM"], PM + 1e-12, color=CM, lw=1.6, label=r"mean moisture $\bar M$")
    axA.semilogy(d["fR"], PR + 1e-12, color=CR, lw=1.6, label=r"surface rate $\partial_\tau M|_0$ (Sobolev target)")
    axA.axvline(float(d["fmaxM"]), color=CM, ls="--", lw=1)
    axA.axvline(float(d["fmaxR"]), color=CR, ls="--", lw=1)
    axA.set_xlim(0, float(d["fmaxR"]) * 2.2); axA.set_ylim(1e-6, 2)
    axA.set_xlabel(r"frequency  $f$  (cycles per unit $\tau$)")
    axA.set_ylabel("normalized power")
    axA.set_title(r"Differentiation tilts the spectrum up by $f$:"
                  "\nthe rate channel is broader (needs finer sampling)", fontsize=10)
    axA.legend(fontsize=8.3)

    # Aliasing: rate-reconstruction error vs sampling interval.
    dtaus, errs = d["dtaus"], d["errs"]
    axB.plot(dtaus, errs * 100, "o-", color=CR, ms=3, lw=1.5)
    nyq = float(d["dtau_R"])
    axB.axvline(nyq, color="k", lw=1.2)
    axB.text(nyq * 1.1, 13.5, f"rate Nyquist\n$\\Delta\\tau^*={nyq:.3f}$", fontsize=8.5)
    axB.axvline(DTAU_PAPER, color="#2ca02c", ls="--", lw=1.4)
    axB.text(DTAU_PAPER * 0.97, 4.5, f"paper sampling\n$\\Delta\\tau\\approx{DTAU_PAPER:.3f}$",
             fontsize=8.5, color="#2ca02c", ha="right")
    axB.axvspan(dtaus.min(), nyq, color="#2ca02c", alpha=0.07)
    axB.axvspan(nyq, dtaus.max(), color=CR, alpha=0.06)
    axB.set_xlabel(r"measurement sampling interval  $\Delta\tau_s$")
    axB.set_ylabel("surface-rate reconstruction error (%)")
    axB.set_title("Under-sampling the derivative target -> aliasing error\n"
                  "(the Sobolev term then fits the artifact)", fontsize=10)

    fig.suptitle("Shannon sampling for the Sobolev constraint: the derivative channel sets a "
                 "stricter Nyquist limit", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
