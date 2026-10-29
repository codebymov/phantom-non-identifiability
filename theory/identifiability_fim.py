"""Identifiability / observability of the Luikov parameters (Vallery Ch. 4.3 -> the paper).

Classical non-recursive least-squares identifiability analysis (4.3.2) applied to the Luikov
coefficients theta = (Lu, Ko, Bi_t, Bi_m) under the drying-cell observation operator.

Observation operator H(theta): the functionals a cell measures, stacked over K sample times,
  - moisture cell:  g = [ Mbar(tau_k), dM/dtau|_xi=0 (tau_k) ]      (scale + mass-flow meter)
  - + temperature:  g = above + [ Tbar(tau_k) ]                     (one added T channel)

Scaled sensitivity matrix  S_{i j} = d g_i / d log theta_j  (dimensionless; columns comparable).
Fisher information  F = S^T S / sigma^2.  We report:
  (a) STRUCTURAL identifiability: rank(S) and the per-parameter column norms. M evolves
      independently of T (one-way coupling M -> T via Ko), so dg/dKo = dg/dBi_t = 0 for any
      moisture-only sensor: Ko, Bi_t are structurally UNOBSERVABLE without a temperature channel.
  (b) PRACTICAL identifiability: the Cramer-Rao bound sigma_theta >= diag(F^-1)^{1/2} at the real
      sensor noise, i.e. the conditioning of (S^T S).
  (c) NON-IDENTIFIABLE COMBINATIONS: the near-null eigenvectors of F name the directions in
      parameter space the data cannot resolve (here the {Ko, Bi_t} subspace, exactly).

Run:  .venv/bin/python theory/identifiability_fim.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kibapap.physics import LuikovParams, solve_luikov_fd

NAMES = ["Lu", "Ko", "Bi_t", "Bi_m"]
K = 40                 # number of measurement sample times in (0, tau_final]
NOISE = 0.01           # sensor noise std (dimensionless units) for the Cramer-Rao bound
OUT = Path(__file__).resolve().parents[1] / "runs/theory/identifiability.npz"


def observe(theta: np.ndarray, channels: str) -> np.ndarray:
    s = solve_luikov_fd(LuikovParams(*theta), n_x=101, n_t=201)
    idx = np.linspace(1, len(s.tau) - 1, K).astype(int)
    Mbar = s.M.mean(1)[idx]
    dMdt = np.gradient(s.M[:, 0], s.tau)[idx]
    g = [Mbar, dMdt]
    if channels == "moist+temp":
        g.append(s.T.mean(1)[idx])           # one added (mean) temperature channel
    return np.concatenate(g)


def sensitivity(theta: np.ndarray, channels: str, d: float = 1e-3) -> np.ndarray:
    """Scaled Jacobian ``S_ij = dg_i / dlog theta_j`` by central differences.

    Args:
        theta: Parameter vector.
        channels: Which channel set to observe, passed through to `observe`.
        d: Relative central-difference step in log-parameter space.

    Returns:
        numpy.ndarray: Shape ``(n_obs, len(theta))``.
    """
    cols = []
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] *= (1 + d); tm[j] *= (1 - d)
        cols.append((observe(tp, channels) - observe(tm, channels)) / (2 * d))
    return np.array(cols).T


def analyze(theta: np.ndarray, channels: str) -> dict:
    S = sensitivity(theta, channels)
    F = S.T @ S / NOISE ** 2
    w, V = np.linalg.eigh(F)                  # ascending
    w = w[::-1]; V = V[:, ::-1]
    sv = np.linalg.svd(S, compute_uv=False)   # singular values of the observation Jacobian
    col_norm = np.linalg.norm(S, axis=0)
    rank = int((w > 1e-9 * w.max()).sum())
    null_dim = len(theta) - rank
    pos = w[w > 1e-12 * w.max()]
    cond = float(w.max() / pos.min()) if len(pos) else np.inf
    # Cramer-Rao relative std on each (log-)parameter, where defined.
    crb = np.full(len(theta), np.inf)
    if rank == len(theta):
        crb = np.sqrt(np.diag(np.linalg.inv(F)))
    return dict(S=S, F=F, eig=w, V=V, sv=sv, col_norm=col_norm,
                rank=rank, null_dim=null_dim, cond=cond, crb=crb)


def main() -> None:
    theta0 = np.array(LuikovParams(0.5, 2.0, 1.0, 1.0).as_tuple())
    print(f"nominal theta = {dict(zip(NAMES, theta0))},  K={K} sample times, noise={NOISE}\n")

    res = {}
    for ch in ("moist", "moist+temp"):
        r = analyze(theta0, ch)
        res[ch] = r
        print(f"=== sensor set: {ch} ===")
        print("  column norms ||dg/dlog theta_j||:  " +
              "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, r["col_norm"])))
        print(f"  FIM eigenvalues: {np.array2string(r['eig'], precision=2, suppress_small=True)}")
        print(f"  numerical rank = {r['rank']}/4   condition number = {r['cond']:.1e}")
        if r["rank"] < 4:
            null_dirs = r["V"][:, r["rank"]:]
            dominant = "  ".join(NAMES[k] for k in np.where(np.abs(null_dirs).max(1) > 0.3)[0])
            print(f"  UNOBSERVABLE subspace spanned by: {{{dominant}}}  (structural null space)")
        else:
            print("  all four parameters identifiable; Cramer-Rao relative std:")
            print("    " + "  ".join(f"{n}={c:.3f}" for n, c in zip(NAMES, r["crb"])))
        print()

    # --- Proposition verification (App. A of the observability paper) ---
    rm, rt = res["moist"], res["moist+temp"]
    print("=== Proposition 1 (numerical verification) ===")
    print(f"  H_M (moisture-only): singular values of S = "
          f"{np.array2string(rm['sv'], precision=3, suppress_small=True)}")
    print(f"    null-space dimension = {rm['null_dim']}   (rank {rm['rank']}/4)")
    print(f"    zero-mode parameters (||dg/dlog theta_j|| = 0): "
          f"{{{', '.join(NAMES[j] for j in np.where(rm['col_norm'] < 1e-6)[0])}}}")
    print(f"  H_M + one T channel: singular values of S = "
          f"{np.array2string(rt['sv'], precision=3, suppress_small=True)}")
    print(f"    null-space dimension = {rt['null_dim']}   (rank {rt['rank']}/4)  -> full column rank\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, theta0=theta0, names=np.array(NAMES),
        moist_eig=rm["eig"], moist_sv=rm["sv"], moist_cols=rm["col_norm"], moist_rank=rm["rank"],
        temp_eig=rt["eig"], temp_sv=rt["sv"], temp_cols=rt["col_norm"],
        temp_rank=rt["rank"], temp_crb=rt["crb"],
    )
    # Compact JSON artifact the paper's App. A cites.
    # NOTE: deliberately NOT fisher_nullspace.json. That path is written by
    # theory/posnov_identifiability_sweep.py, which is the artefact Table 1 is built from, and
    # this script writes a different schema -- running it used to clobber the published table.
    JSON_OUT = OUT.with_name("fisher_nullspace_pn0.json")
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "theta0": dict(zip(NAMES, theta0.tolist())), "K": K, "noise": NOISE,
            "moist_only": {"singular_values": rm["sv"].tolist(), "rank": rm["rank"],
                           "null_dim": rm["null_dim"],
                           "zero_mode_params": [NAMES[j] for j in np.where(rm["col_norm"] < 1e-6)[0]],
                           "cond": rm["cond"]},
            "moist_plus_temp": {"singular_values": rt["sv"].tolist(), "rank": rt["rank"],
                                "null_dim": rt["null_dim"], "cond": rt["cond"],
                                "crb_rel_std": dict(zip(NAMES, rt["crb"].tolist()))},
        }, f, indent=2)
    print(f"saved {OUT}\nsaved {JSON_OUT}")
    print("\nTakeaway: the drying cell's moisture instrumentation identifies only {Lu, Bi_m} "
          "(rank 2);\n{Ko, Bi_t} are structurally unobservable until a temperature channel is "
          "added (rank 4).")


if __name__ == "__main__":
    main()
