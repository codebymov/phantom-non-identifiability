"""The B<->C bridge: when do the Sobolev rows add information rather than aliasing variance?

App. B treats the surface-rate rows of the sensitivity matrix S as added Fisher information:
F_Sob = F_plain + S_rate^T Sigma_rate^{-1} S_rate >= F_plain (Loewner). That presumes the rate
channel is measured faithfully. Under-sampling (App. C) replaces the true functional by a
finite-difference estimate carrying a DETERMINISTIC aliasing bias a(theta; dtau). The realized
(Godambe / mean-squared-error) information then is

    F_Sob^eff(dtau) = MSE_Sob(dtau)^{-1},
    MSE_Sob = (S^T Sig^{-1} S)^{-1} + b_theta b_theta^T,   b_theta = (S^T Sig^{-1} S)^{-1} S^T Sig^{-1} b,

with S = [S_mean; S_rate], Sig = blkdiag(Sigma_mean, Sigma_rate), b = [0; a]. F_Sob^eff >= F_plain
(PSD gap) iff the alias term a vanishes, i.e. iff dtau < dtau* = 0.022 (rate Nyquist, App. C).
Below threshold the bias inflates MSE_Sob and F_Sob^eff - F_plain acquires a negative eigenvalue:
the Sobolev rows then inject aliasing variance, not information. We locate the crossover on the
identifiable subspace theta = (Lu, Bi_m) and evaluate it at our schedule dtau ~ 0.05.

Run:  .venv/bin/python theory/sobolev_information_crossover.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kibapap.physics import LuikovParams, solve_luikov_fd

OUT = Path(__file__).resolve().parents[1] / "runs/theory/sobolev_crossover.npz"
TAU0, TAUF = 0.05, 2.0          # restrict past the singular step-BC onset (App. C)
TH0 = np.array([0.5, 1.0])      # identifiable subspace (Lu, Bi_m); Ko, Bi_t fixed
KO, BIT = 2.0, 1.0
SIG_M, SIG_R = 0.01, 0.01       # mean / rate sensor noise (dimensionless)
DTAU_STAR = 0.022               # rate Nyquist from App. C
DTAU_PAPER = 2.0 / 40


def fine():
    s = solve_luikov_fd(LuikovParams(TH0[0], KO, BIT, TH0[1]), n_x=201, n_t=4001, tau_final=TAUF)
    return s.tau, s.M.mean(1), s.M[:, 0]


def signals(theta, tau_q):
    """Mean moisture and the true surface drying rate, interpolated onto query times.

    Args:
        theta: ``(Lu, Bi_m)``; Ko and Bi_t are held at module constants.
        tau_q: Query times.

    Returns:
        tuple: ``(Mbar, rate, Msurf)``, each of shape ``(len(tau_q),)``. Computed on a finer
            grid than the query schedule so the rate is the true derivative rather than a
            difference of the sampled trace.
    """
    s = solve_luikov_fd(LuikovParams(theta[0], KO, BIT, theta[1]), n_x=201, n_t=4001, tau_final=TAUF)
    Mbar = np.interp(tau_q, s.tau, s.M.mean(1))
    rate = np.interp(tau_q, s.tau, np.gradient(s.M[:, 0], s.tau))
    Msurf = np.interp(tau_q, s.tau, s.M[:, 0])
    return Mbar, rate, Msurf


def fd_rate(Msurf, tau_q):
    return np.gradient(Msurf, tau_q)         # what the loss actually uses as the rate target


def channels(theta, tau_q):
    Mbar, rate_true, Msurf = signals(theta, tau_q)
    return Mbar, fd_rate(Msurf, tau_q), rate_true


def jac(fn, tau_q, d=2e-3):
    cols = []
    for j in range(2):
        tp, tm = TH0.copy(), TH0.copy(); tp[j] *= (1 + d); tm[j] *= (1 - d)
        cols.append((fn(tp, tau_q) - fn(tm, tau_q)) / (TH0[j] * 2 * d))
    return np.array(cols).T          # (K, 2)  -- absolute sensitivity d g / d theta_j


def analyze(dtau):
    tau_q = np.arange(TAU0, TAUF + 1e-9, dtau)
    if len(tau_q) < 6:
        return None
    Sm = jac(lambda th, t: channels(th, t)[0], tau_q)
    Sr = jac(lambda th, t: channels(th, t)[1], tau_q)
    # aliasing bias of the rate target at theta0: finite-diff estimate minus the true rate.
    _, rfd, rtrue = channels(TH0, tau_q)
    a = rfd - rtrue
    # plain (mean-only) and Sobolev (mean+rate) realized information on (Lu, Bi_m).
    Fm = (Sm.T @ Sm) / SIG_M ** 2
    S = np.vstack([Sm, Sr]); Sig_inv = np.diag(np.r_[np.full(len(Sm), 1/SIG_M**2),
                                                      np.full(len(Sr), 1/SIG_R**2)])
    A = np.linalg.inv(S.T @ Sig_inv @ S)
    b = np.r_[np.zeros(len(Sm)), a]
    bias_th = A @ S.T @ Sig_inv @ b
    MSE_Sob = A + np.outer(bias_th, bias_th)
    F_Sob_eff = np.linalg.inv(MSE_Sob)
    gap = F_Sob_eff - Fm                  # F_Sob - F_plain (realized); PSD iff alias-free
    return float(np.linalg.eigvalsh(gap).min()), float(np.linalg.norm(a) / np.linalg.norm(rtrue))


def main():
    dtaus = np.unique(np.r_[np.linspace(0.004, 0.09, 60), DTAU_STAR, DTAU_PAPER])
    mineig, alias = [], []
    for dt in dtaus:
        r = analyze(dt)
        mineig.append(np.nan if r is None else r[0]); alias.append(np.nan if r is None else r[1])
    mineig = np.array(mineig); alias = np.array(alias)
    valid = ~np.isnan(mineig)
    sign = np.sign(mineig[valid])
    flips = sign[:-1] != sign[1:]
    cross = dtaus[valid][np.argmax(flips)] if flips.any() else np.nan
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, dtaus=dtaus, mineig=mineig, alias=alias,
                        crossover=cross, dtau_star=DTAU_STAR, dtau_paper=DTAU_PAPER)

    def at(dt):
        i = int(np.nanargmin(np.abs(dtaus - dt))); return mineig[i], alias[i]
    print(f"identifiable subspace theta=(Lu,Bi_m); rate Nyquist dtau*={DTAU_STAR}\n")
    print(f"  {'dtau':>8} {'min eig(F_Sob - F_plain)':>26} {'rate alias error':>18}")
    for dt in [0.006, DTAU_STAR, 0.035, DTAU_PAPER, 0.07]:
        me, al = at(dt); print(f"  {dt:8.3f} {me:26.4f} {al*100:16.1f}%")
    print(f"\nLoewner crossover (min-eig -> 0): dtau ~ {cross:.3f}  (rate Nyquist {DTAU_STAR})")
    me, al = at(DTAU_PAPER)
    verdict = "INDEFINITE (Sobolev rows inject aliasing variance)" if me < 0 else "still PSD"
    print(f"at our schedule dtau={DTAU_PAPER:.3f} ({al*100:.0f}% alias): "
          f"min-eig={me:.3f} -> F_Sob - F_plain {verdict}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
