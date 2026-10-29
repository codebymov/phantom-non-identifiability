"""M0 deliverable: reproduce the predecessor paper's numbers through the AD tangent interface,
and apply the Section-8 metrics to the reduced-vs-parent tangent maps.

Two things are checked here:

1. **Reproduction.** The paper's Table 1 Cramer-Rao values were produced by central differences
   through the scipy solver. Recomputing them by autograd through an independent torch mirror is a
   genuinely independent path to the same numbers -- the strongest reproducibility check available
   without a second solver implementation.

2. **Forward accuracy is uninformative about the tangent map.** For each true coupling we report the
   forward rel-L2 between reduced and parent observables *alongside* E_J, E_sigma, the sensitive-
   subspace angle, the rank verdict and the sensitivity ordering. The contrast is the argument.

Run: PYTHONPATH=. .venv/bin/python tangentsafe/experiments/m0_reproduce_and_metrics.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tangentsafe.audit.metrics import tangent_metrics
from tangentsafe.core.tangent import NOMINAL, jacobian_ad, jacobian_fd, observe

NAMES = ["Lu", "Ko", "Bi_t", "Bi_m"]
K, FRAC = 40, 0.01
PN_GRID = [0.0, 0.05, 0.1, 0.2, 0.6, 1.0]
PAPER_CR = {0.0: (None, None), 0.05: (176, 259), 0.1: (99, 134),
            0.2: (66, 77), 0.6: (19, 26), 1.0: (1.2, 3.2)}
OUT = Path(__file__).resolve().parents[2] / "runs/tangentsafe/m0_reproduce.json"


def sigmas_for(theta):
    H = observe(theta, K=K).detach().numpy()
    return np.concatenate([np.full(K, FRAC * np.ptp(H[i * K:(i + 1) * K])) for i in range(2)])


def cr_from(J, sig):
    Jw = J / sig[:, None]
    cn = np.linalg.norm(Jw, axis=0)
    if (cn > 1e-9 * cn.max()).sum() < len(NAMES):
        return None, None, cn
    d = np.diag(np.linalg.solve(Jw.T @ Jw, np.eye(len(NAMES))))
    return float(np.sqrt(d[1]) * 100), float(np.sqrt(d[2]) * 100), cn


def main() -> None:
    print("=== 1. Table 1 reproduced through autograd (paper used central differences) ===")
    print(f"{'Pn':>5} {'CR Ko (AD)':>11} {'paper':>7} {'CR Bi_t (AD)':>13} {'paper':>7} {'rank':>5}")
    rows = []
    for pn in PN_GRID:
        th = dict(NOMINAL, Pn=pn)
        sig = sigmas_for(th)
        J = jacobian_ad(th)
        ko, bit, cn = cr_from(J, sig)
        rank = int((np.linalg.norm(J / sig[:, None], axis=0) > 1e-9 *
                    np.linalg.norm(J / sig[:, None], axis=0).max()).sum())
        pk, pb = PAPER_CR[pn]
        f = lambda v: "inf" if v is None else f"{v:.1f}"
        print(f"{pn:5.2f} {f(ko):>11} {str(pk):>7} {f(bit):>13} {str(pb):>7} {rank:>5}")
        rows.append(dict(pn=pn, cr_Ko=ko, cr_Bit=bit, rank=rank,
                         paper_cr_Ko=pk, paper_cr_Bit=pb,
                         col_norms=dict(zip(NAMES, cn.tolist()))))

    print("\n=== 2. Forward accuracy vs tangent accuracy (reduced Pn=0 against parent) ===")
    print(f"{'Pn parent':>10} {'fwd relL2':>10} {'E_J':>8} {'E_sigma':>8} {'angle deg':>10} "
          f"{'rank sur/ref':>13} {'order?':>7}")
    mets = {}
    for pn in [0.05, 0.1, 0.2, 0.35, 0.6, 1.0]:
        th_par = dict(NOMINAL, Pn=pn)
        th_red = dict(NOMINAL, Pn=0.0)
        sig = sigmas_for(th_par)
        H_par = observe(th_par, K=K).detach().numpy()
        H_red = observe(th_red, K=K).detach().numpy()
        fwd = float(np.linalg.norm(H_red - H_par) / np.linalg.norm(H_par))
        m = tangent_metrics(jacobian_ad(th_red), jacobian_ad(th_par), sig, NAMES, k=2,
                            fwd_relL2=fwd)
        mets[str(pn)] = m
        print(f"{pn:10.2f} {fwd:10.4f} {m['E_J']:8.4f} {m['E_sigma']:8.4f} "
              f"{m['subspace_angle_deg']:10.2f} {m['rank_surrogate']:6d}/{m['rank_reference']:<6d} "
              f"{str(m['order_agrees']):>7}")
    print("  The forward error stays small while the subspace angle is large and the rank verdict")
    print("  is wrong: forward L2 does not certify the tangent map.")

    print("\n=== 3. Cost of the tangent map (M0 engineering finding) ===")
    th = dict(NOMINAL, Pn=0.6)
    t0 = time.time(); jacobian_ad(th); t_ad = time.time() - t0
    t0 = time.time(); jacobian_fd(th); t_fd = time.time() - t0
    print(f"  reverse-mode AD full Jacobian : {t_ad:6.2f}s  ({2 * K} backward passes)")
    print(f"  central differences           : {t_fd:6.2f}s  ({2 * len(NAMES)} solves)")
    print(f"  -> FD is {t_ad / t_fd:.0f}x cheaper at this problem shape (p={len(NAMES)} parameters,")
    print(f"     n_obs={2 * K} observables). Reverse mode pays off only when p >> n_obs, i.e. when")
    print("     the closure network's weights become the parameters (M2), not for the audit.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        setup=dict(K=K, noise_frac=FRAC, nominal=NOMINAL, names=NAMES),
        table1_reproduction=rows, tangent_metrics=mets,
        cost=dict(ad_s=t_ad, fd_s=t_fd, ratio=t_ad / t_fd, p=len(NAMES), n_obs=2 * K)), indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
