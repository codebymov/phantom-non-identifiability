"""Shannon sampling applied to the measurement-consistency / Sobolev loss (Vallery Ch. 4.5).

The Sobolev term constrains a DERIVATIVE functional -- the surface drying rate
r(tau) = dM/dtau|_xi=0 -- against measurements sampled at spacing dtau_s. Differentiation
multiplies the spectrum by (i 2 pi f), so the derivative channel has a HIGHER effective
bandwidth than the mean-moisture channel: the Sobolev constraint needs a finer sampling rate
than a plain data fit. Below the Nyquist rate the sampled-derivative target is aliased, and the
Sobolev term then fits the aliasing artifact rather than the physical gradient.

We (1) compute the spectra and effective bandwidths f_max of the mean and rate channels from a
fine FD solution, (2) derive the Nyquist spacing dtau* = 1/(2 f_max), and (3) demonstrate that
the reconstruction error of r(tau) from samples grows sharply once dtau_s > dtau*.

Run:  .venv/bin/python theory/sampling_nyquist.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kibapap.physics import LuikovParams, solve_luikov_fd

OUT = Path(__file__).resolve().parents[1] / "runs/theory/sampling.npz"
ENERGY = 0.95   # bandwidth = smallest f holding this fraction of spectral energy
# (95% rather than 99% so the estimate reflects the resolved dynamics, not the grid-limited
#  high-frequency tail of the near-singular initial surface transient.)


def bandwidth(sig: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
    sig = sig - sig.mean()
    F = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), dt)
    P = np.abs(F) ** 2
    cum = np.cumsum(P) / P.sum()
    f_max = float(f[np.searchsorted(cum, ENERGY)])
    return f, P, f_max


def main() -> None:
    p = LuikovParams(0.5, 2.0, 1.0, 1.0)
    s = solve_luikov_fd(p, n_x=201, n_t=4001, tau_final=2.0)   # fine reference
    # Restrict to tau >= TAU0: the step-BC surface rate has a t^{-1/2}-type singular onset whose
    # bandwidth is grid-limited and unphysical; past the onset the dynamics are smooth and the
    # bandwidth reflects the resolved diffusion modes (a real cell ramps the air over a finite time).
    TAU0 = 0.05
    m0 = s.tau >= TAU0
    tau = s.tau[m0]; dt = tau[1] - tau[0]
    Mbar = s.M.mean(1)[m0]
    rate = np.gradient(s.M[:, 0], s.tau)[m0]                    # surface drying rate (Sobolev target)

    fM, PM, fmaxM = bandwidth(Mbar, dt)
    fR, PR, fmaxR = bandwidth(rate, dt)
    dtau_M, dtau_R = 1 / (2 * fmaxM), 1 / (2 * fmaxR)
    print(f"effective bandwidth (99% energy):  mean Mbar f_max={fmaxM:.2f}  ->  dtau* = {dtau_M:.4f}")
    print(f"                                    rate dM/dtau f_max={fmaxR:.2f}  ->  dtau* = {dtau_R:.4f}")
    print(f"  the derivative channel needs ~{dtau_M/dtau_R:.1f}x finer sampling than the mean.\n")

    # Aliasing demonstration: estimate the rate from sub-sampled surface moisture and compare
    # to truth. Error stays at the truncation floor while dtau_s < dtau*_rate, then blows up.
    Msurf = s.M[:, 0][m0]
    rate_true = rate
    dtaus, errs = [], []
    for step in range(1, 120):
        ts = tau[::step]
        if len(ts) < 8:
            break
        r_s = np.gradient(Msurf[::step], ts)
        r_ref = np.interp(ts, tau, rate_true)
        err = np.linalg.norm(r_s - r_ref) / np.linalg.norm(r_ref)
        dtaus.append(ts[1] - ts[0]); errs.append(float(err))
    dtaus = np.array(dtaus); errs = np.array(errs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, fM=fM, PM=PM, fR=fR, PR=PR, fmaxM=fmaxM, fmaxR=fmaxR,
                        dtau_M=dtau_M, dtau_R=dtau_R, dtaus=dtaus, errs=errs)
    # Report the error at, just below, and well above Nyquist.
    for label, target in [("dtau* (rate Nyquist)", dtau_R), ("2x Nyquist", 2 * dtau_R),
                          ("4x Nyquist", 4 * dtau_R)]:
        i = int(np.argmin(np.abs(dtaus - target)))
        print(f"  rate reconstruction rel-error at {label:22s} (dtau={dtaus[i]:.3f}): {errs[i]*100:.1f}%")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
