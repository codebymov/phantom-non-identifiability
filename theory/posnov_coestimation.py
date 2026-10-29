"""Does moisture-only recovery of Ko survive co-estimating the Posnov number?

Reviewer concern: Pn is held fixed but is itself uncertain (about 0.2 to 2), and both Ko and
Pn scale the T-M coupling, so they might trade off. We extend the Fisher analysis to the
5-parameter vector (Lu, Ko, Bi_t, Bi_m, Pn) at a realistic Pn=0.6 and report the Cramer-Rao
relative std and the Ko-Pn correlation, for moisture-only and moisture+temperature sensors.

Result (moisture-only): full rank; CR(Ko) 19% -> 20%; corr(Ko,Pn)=+0.35 -> no degeneracy.

Run:  .venv/bin/python theory/posnov_coestimation.py
"""
from __future__ import annotations

import numpy as np

from kibapap.physics import LuikovParams, solve_luikov_fd

K, NOISE, D = 40, 0.01, 1e-3
NAMES = ["Lu", "Ko", "Bi_t", "Bi_m", "Pn"]
TH = np.array([0.5, 2.0, 1.0, 1.0, 0.6])   # realistic paper Pn

def observe(t, channels):
    s = solve_luikov_fd(LuikovParams(t[0], t[1], t[2], t[3], Pn=t[4]), n_x=101, n_t=201)
    idx = np.linspace(1, len(s.tau) - 1, K).astype(int)
    g = [s.M.mean(1)[idx], np.gradient(s.M[:, 0], s.tau)[idx]]
    if channels == "moist+temp":
        g.append(s.T.mean(1)[idx])
    return np.concatenate(g)

def analyze(channels, npar):
    cols = []
    for j in range(npar):
        tp, tm = TH.copy(), TH.copy(); tp[j] *= (1 + D); tm[j] *= (1 - D)
        cols.append((observe(tp, channels) - observe(tm, channels)) / (2 * D))
    S = np.array(cols).T; F = S.T @ S / NOISE ** 2
    w = np.linalg.eigvalsh(F)[::-1]; rank = int((w > 1e-9 * w.max()).sum())
    if rank < npar:
        return None, rank, None
    Fi = np.linalg.inv(F); cr = np.sqrt(np.diag(Fi)) * 100
    d = np.sqrt(np.diag(Fi)); corr = Fi / np.outer(d, d)
    return cr, rank, corr

def main():
    for ch in ["moist", "moist+temp"]:
        cr4, r4, _ = analyze(ch, 4)
        cr5, r5, corr = analyze(ch, 5)
        print(f"--- {ch} ---")
        print(f"  4-param (Pn known):     rank {r4}/4  " +
              "  ".join(f"{n}={c:.1f}%" for n, c in zip(NAMES[:4], cr4)))
        print(f"  5-param (Pn estimated): rank {r5}/5  " +
              "  ".join(f"{n}={c:.1f}%" for n, c in zip(NAMES, cr5)))
        i, j = NAMES.index("Ko"), NAMES.index("Pn")
        print(f"  corr(Ko,Pn)={corr[i,j]:+.3f}   CR(Ko): {cr4[1]:.1f}% -> {cr5[1]:.1f}%\n")

if __name__ == "__main__":
    main()
