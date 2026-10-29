"""OSSE: the phantom identifiability boundary in a parameter-recovery experiment.

Truth is a fraternal-twin FD solve at a realistic paper Posnov number (Pn_true=0.6).
From the two moisture-only sensors a drying cell provides (mean moisture + surface rate,
noisy), we recover the pair {Ko, Bi_t} that the Pn=0 Proposition calls "structurally
unobservable", using solver-in-the-loop least squares with two forward models:

  - FULL     : Pn_model = 0.6  (correctly specified) -> should recover {Ko, Bi_t}
  - REDUCED  : Pn_model = 0.0  (the standard simplification) -> {Ko,Bi_t} do not affect
               the moisture sensors, so the estimator cannot move off its prior.

Fraternal twin (truth 201x401, model 101x201) avoids the inverse crime.
Run:  .venv/bin/python theory/posnov_osse.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from kibapap.physics import LuikovParams, solve_luikov_fd

THETA_TRUE = dict(Lu=0.5, Ko=2.0, Bi_t=1.0, Bi_m=1.0)
PN_TRUE = 0.6
K = 40
FRAC = 0.01                 # sensor noise as fraction of each channel's range
SEEDS = [0, 1, 2, 3, 4]
KO0, BIT0 = 1.0, 0.5        # wrong prior for the unknowns

def sensors(sol, t_samp):
    Mbar = np.interp(t_samp, sol.tau, sol.M.mean(1))
    rate = np.interp(t_samp, sol.tau, np.gradient(sol.M[:, 0], sol.tau))
    return Mbar, rate

def main(truth_grid=(101, 201)):
    # truth_grid=(101,201) -> matched-grid identifiability check (isolates the Fisher verdict);
    # truth_grid=(201,401) -> fraternal twin (adds discretisation/model-form error).
    truth = solve_luikov_fd(LuikovParams(**THETA_TRUE, Pn=PN_TRUE), n_x=truth_grid[0], n_t=truth_grid[1])
    idx = np.linspace(1, len(truth.tau) - 1, K).astype(int)
    t_samp = truth.tau[idx]
    Mbar_t, rate_t = sensors(truth, t_samp)
    sM = FRAC * (Mbar_t.max() - Mbar_t.min()); sR = FRAC * (rate_t.max() - rate_t.min())

    def resid(logp, yM, yR, pn_model):
        Ko, Bit = np.exp(logp)
        s = solve_luikov_fd(LuikovParams(Lu=THETA_TRUE["Lu"], Ko=Ko, Bi_t=Bit,
                                         Bi_m=THETA_TRUE["Bi_m"], Pn=pn_model), n_x=101, n_t=201)
        Mb, rt = sensors(s, t_samp)
        return np.r_[(Mb - yM) / sM, (rt - yR) / sR]

    def field_err(Ko, Bit, pn_model):
        s = solve_luikov_fd(LuikovParams(Lu=THETA_TRUE["Lu"], Ko=Ko, Bi_t=Bit,
                                         Bi_m=THETA_TRUE["Bi_m"], Pn=pn_model),
                            n_x=truth_grid[0], n_t=truth_grid[1])
        return float(np.linalg.norm(s.M - truth.M) / np.linalg.norm(truth.M))

    out = {}
    for pn_model, name in [(PN_TRUE, "FULL (Pn=0.6)"), (0.0, "REDUCED (Pn=0)")]:
        Kos, Bits, ferr = [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            yM = Mbar_t + rng.normal(0, sM, Mbar_t.shape)
            yR = rate_t + rng.normal(0, sR, rate_t.shape)
            r = least_squares(resid, np.log([KO0, BIT0]), args=(yM, yR, pn_model),
                              method="lm", max_nfev=200)
            Ko, Bit = np.exp(r.x)
            Kos.append(Ko); Bits.append(Bit); ferr.append(field_err(Ko, Bit, pn_model))
        Kos, Bits, ferr = map(np.array, (Kos, Bits, ferr))
        out[name] = (Kos, Bits, ferr)
        eKo = np.abs(Kos.mean() - THETA_TRUE["Ko"]) / THETA_TRUE["Ko"] * 100
        eBit = np.abs(Bits.mean() - THETA_TRUE["Bi_t"]) / THETA_TRUE["Bi_t"] * 100
        print(f"{name:16s}: Ko={Kos.mean():.3f}±{Kos.std():.3f} (err {eKo:4.1f}%, true 2.0) | "
              f"Bi_t={Bits.mean():.3f}±{Bits.std():.3f} (err {eBit:4.1f}%, true 1.0) | "
              f"moisture rel-L2={ferr.mean():.4f}")
    print(f"\nprior (wrong): Ko={KO0}, Bi_t={BIT0}  ;  truth Ko=2.0, Bi_t=1.0, Pn={PN_TRUE}")

if __name__ == "__main__":
    print("=== matched-grid identifiability check (truth 101x201) ===")
    main(truth_grid=(101, 201))
    print("\n=== fraternal-twin robustness (truth 201x401, model 101x201) ===")
    main(truth_grid=(201, 401))
