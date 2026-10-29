"""E1: can any loss or architecture restore a severed pathway? (Proposition 1).

The reviewer objection this forecloses: "surely gradient-enhanced penalties, better loss weighting,
or an architecture that feeds the temperature field to the moisture head would fix this."

Proposition 1 says no. If the residual is built from the reduced operator, in which Ko and Bi_t
appear only in the temperature equation and the moisture equation is autonomous, then
dH_M/dKo == 0 as an identity of the computational graph -- for every parameterisation, every loss
weighting, every training outcome. A loss can only reweight paths that exist.

Five arms, all trained on the SAME reduced (Pn=0) physics, differing only in objective or
architecture:

  baseline    plain physics-informed residual (PDE + BC), hard initial conditions
  gpinn       + gradient-enhanced residual penalty: ||d r / d xi||^2
  adaptive    + gradient-norm loss balancing across residual and BC terms
  causal      + causal temporal weighting (early times weighted first)
  sharedtrunk architecture change: shared trunk, and the moisture head is explicitly fed the
              temperature head's output and its spatial gradient -- the most favourable
              architecture for the pathway to reappear

Measured per arm: forward accuracy against the FD solver, and the scaled sensitivity
|d<M>/dlog Ko| by autograd, alongside |d<T>/dlog Ko| for the pathway that is intact. The ratio of
the two is the leakage floor: severed-pathway sensitivity relative to a pathway the same network
represents well.

Run: PYTHONPATH=. .venv/bin/python tangentsafe/experiments/e1_objective_impossibility.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from kibapap.physics import LuikovParams, solve_luikov_fd

torch.set_default_dtype(torch.float64)
SEED = 0
SEEDS = [0, 1, 2]
BOX = dict(Bi_t=(0.5, 2.0), Bi_m=(0.5, 2.0), Lu=(0.2, 1.0), Ko=(1.0, 3.0))
ORDER = ["Bi_t", "Bi_m", "Lu", "Ko"]
NOMINAL = dict(Bi_t=1.0, Bi_m=1.0, Lu=0.5, Ko=2.0)
N_COLL, ITERS, WIDTH, DEPTH = 4000, 12000, 96, 5
LBFGS_ITERS = 800
OUT = Path(__file__).resolve().parents[2] / "runs/tangentsafe/e1_impossibility.json"


class Net(torch.nn.Module):
    """Two heads over (xi, tau, Bi_t, Bi_m, Lu, Ko). Hard ICs: T = tau*f, M = 1 + tau*g."""

    def __init__(self, shared_trunk=False, feed_T_to_M=False, w=WIDTH, d=DEPTH):
        super().__init__()
        self.shared, self.feed = shared_trunk, feed_T_to_M
        mk = lambda i, o: torch.nn.Sequential(
            *[layer for k in range(d)
              for layer in (torch.nn.Linear(i if k == 0 else w, w), torch.nn.Tanh())],
            torch.nn.Linear(w, o))
        if shared_trunk:
            self.trunk = mk(6, w)
            self.hT = torch.nn.Linear(w, 1)
            self.hM = torch.nn.Sequential(torch.nn.Linear(w + (2 if feed_T_to_M else 0), w),
                                          torch.nn.Tanh(), torch.nn.Linear(w, 1))
        else:
            self.netT, self.netM = mk(6, 1), mk(6, 1)

    def forward(self, x):
        tau = x[:, 1:2]
        if not self.shared:
            return tau * self.netT(x), 1.0 + tau * self.netM(x)
        h = self.trunk(x)
        T = tau * self.hT(h)
        if self.feed:
            dTdxi = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 0:1]
            h = torch.cat([h, T, dTdxi], dim=1)
        return T, 1.0 + tau * self.hM(h)


def sample(n, rng):
    cols = [rng.uniform(0, 1, n), rng.uniform(0, 2, n)] + [rng.uniform(*BOX[k], n) for k in ORDER]
    return torch.tensor(np.stack(cols, -1), requires_grad=True)


def residuals(net, x):
    T, M = net(x)
    g = lambda y: torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    dT, dM = g(T), g(M)
    dTx, dTt, dMx, dMt = dT[:, 0:1], dT[:, 1:2], dM[:, 0:1], dM[:, 1:2]
    dTxx = torch.autograd.grad(dTx.sum(), x, create_graph=True)[0][:, 0:1]
    dMxx = torch.autograd.grad(dMx.sum(), x, create_graph=True)[0][:, 0:1]
    Bi_t, Bi_m, Lu, Ko = x[:, 2:3], x[:, 3:4], x[:, 4:5], x[:, 5:6]
    rT = dTt - dTxx - Ko * dMt                    # REDUCED physics: M is autonomous
    rM = dMt - Lu * dMxx
    b0 = (x[:, 0:1] < 0.02).double()
    b1 = (x[:, 0:1] > 0.98).double()
    rB = b0 * ((dTx - Bi_t * (T - 1.0)) ** 2 + (dMx - Bi_m * M) ** 2) + b1 * (dTx ** 2 + dMx ** 2)
    return rT, rM, rB


def train(arm, seed=SEED):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(shared_trunk=(arm == "sharedtrunk"), feed_T_to_M=(arm == "sharedtrunk"))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lam_bc = 1.0
    for it in range(ITERS):
        x = sample(N_COLL, rng)
        rT, rM, rB = residuals(net, x)
        L_pde = (rT ** 2).mean() + (rM ** 2).mean()
        L_bc = rB.mean()
        if arm == "causal":
            w = torch.exp(-3.0 * x[:, 1:2] * max(0.0, 1 - it / ITERS))
            L_pde = ((w * rT ** 2).mean() + (w * rM ** 2).mean()) / w.mean()
        if arm == "gpinn":
            grT = torch.autograd.grad(rT.sum(), x, create_graph=True)[0][:, 0:1]
            grM = torch.autograd.grad(rM.sum(), x, create_graph=True)[0][:, 0:1]
            L_pde = L_pde + 1e-2 * ((grT ** 2).mean() + (grM ** 2).mean())
        if arm == "adaptive" and it % 100 == 0:
            ps = [p for p in net.parameters()]
            gp = torch.autograd.grad(L_pde, ps, retain_graph=True, allow_unused=True)
            gb = torch.autograd.grad(L_bc, ps, retain_graph=True, allow_unused=True)
            npde = sum(float((g ** 2).sum()) for g in gp if g is not None) ** 0.5
            nbc = sum(float((g ** 2).sum()) for g in gb if g is not None) ** 0.5
            lam_bc = float(np.clip(npde / max(nbc, 1e-12), 1e-2, 1e2))
        loss = L_pde + lam_bc * L_bc
        opt.zero_grad(); loss.backward(); opt.step()

    # L-BFGS polish on a FIXED collocation set, matching the protocol of the paper's own surrogate
    # (12k Adam then L-BFGS). Without it the arms are under-converged relative to that surrogate,
    # and the measured leakage reports optimisation residue rather than the structural limit.
    xf = sample(N_COLL, rng)
    lb = torch.optim.LBFGS(net.parameters(), max_iter=LBFGS_ITERS, history_size=50,
                           tolerance_grad=1e-12, tolerance_change=1e-14,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        rT, rM, rB = residuals(net, xf)
        L = (rT ** 2).mean() + (rM ** 2).mean() + lam_bc * rB.mean()
        L.backward()
        return L

    final = float(lb.step(closure).detach())
    return net, final


def mean_fields(net, theta, K=40):
    xi = torch.linspace(0, 1, 51); tau = torch.linspace(0.05, 2.0, K)
    XI, TAU = torch.meshgrid(xi, tau, indexing="ij")
    n = XI.numel()
    cols = [XI.reshape(-1), TAU.reshape(-1)] + [theta[k].expand(n) for k in ORDER]
    x = torch.stack(cols, -1).requires_grad_(True)
    T, M = net(x)
    return M.reshape(len(xi), K).mean(0), T.reshape(len(xi), K).mean(0)


def sensitivity(net, name="Ko"):
    """||d<M>/dlog Ko|| and ||d<T>/dlog Ko||: L2 norms of the per-time sensitivity VECTORS.

    Matches the predecessor paper's convention. An earlier version differentiated a summed scalar,
    which permits sign cancellation across time and is not comparable to the solver reference.

    Args:
        net: The trained network to probe.
        name: Parameter to differentiate with respect to. Defaults to ``"Ko"``.

    Returns:
        tuple: ``(||d<M>/dlog p||, ||d<T>/dlog p||)``. The first is the severed pathway, whose
            true value under the reduced operator is exactly zero, so what it measures on a
            finitely-trained network is a leakage floor rather than a coupling.
    """
    theta = {k: torch.tensor(NOMINAL[k], requires_grad=(k == name)) for k in ORDER}
    Mbar, Tbar = mean_fields(net, theta)
    K = Mbar.numel()
    eye = torch.eye(K)
    gM = torch.autograd.grad(Mbar, theta[name], grad_outputs=eye, retain_graph=True,
                             is_grads_batched=True, allow_unused=True)[0]
    gT = torch.autograd.grad(Tbar, theta[name], grad_outputs=eye, retain_graph=True,
                             is_grads_batched=True, allow_unused=True)[0]
    f = lambda v: 0.0 if v is None else float(torch.linalg.norm(v.reshape(-1))) * NOMINAL[name]
    return f(gM), f(gT)


def fd_reference():
    out = {}
    for pn in (0.0, 0.6):
        d, vals = 1e-3, []
        for s in (+1, -1):
            sol = solve_luikov_fd(LuikovParams(NOMINAL["Lu"], NOMINAL["Ko"] * (1 + s * d),
                                               NOMINAL["Bi_t"], NOMINAL["Bi_m"], Pn=pn),
                                  n_x=101, n_t=201)
            idx = np.linspace(1, len(sol.tau) - 1, 40).astype(int)
            vals.append(sol.M.mean(1)[idx])
        out[str(pn)] = float(np.linalg.norm((vals[0] - vals[1]) / (2 * d)) * NOMINAL["Ko"])
    return out


def main() -> None:
    ref = fd_reference()
    print("solver reference, ||d<M>/dlog Ko|| :")
    print(f"  reduced model (Pn=0) : {ref['0.0']:.3e}   <- exactly zero, by construction")
    print(f"  true model  (Pn=0.6) : {ref['0.6']:.3e}\n")
    rows = {}
    sol = solve_luikov_fd(LuikovParams(NOMINAL["Lu"], NOMINAL["Ko"], NOMINAL["Bi_t"],
                                       NOMINAL["Bi_m"], Pn=0.0), n_x=101, n_t=201)
    tau = np.linspace(0.05, 2.0, 40)
    ref_M = np.interp(tau, sol.tau, sol.M.mean(1))
    hdr = (f"{'arm':>12} {'fwd relL2':>16} {'|dM/dlogKo|':>18} {'leakage %':>16}")
    print(hdr); print("-" * len(hdr))
    for arm in ("baseline", "gpinn", "adaptive", "causal", "sharedtrunk"):
        per = []
        for sd in SEEDS:
            t0 = time.time(); net, loss = train(arm, seed=sd); dt = time.time() - t0
            dM, dT = sensitivity(net)
            theta = {k: torch.tensor(NOMINAL[k]) for k in ORDER}
            Mbar, _ = mean_fields(net, theta)
            fwd = float(np.linalg.norm(Mbar.detach().numpy() - ref_M) / np.linalg.norm(ref_M))
            per.append(dict(seed=sd, loss=loss, fwd_relL2=fwd, dM_dlogKo=dM, dT_dlogKo=dT,
                            ratio=dM / max(dT, 1e-30), train_s=dt))
        agg = lambda k: (float(np.mean([q[k] for q in per])), float(np.std([q[k] for q in per])))
        f_m, f_s = agg("fwd_relL2"); m_m, m_s = agg("dM_dlogKo"); r_m, r_s = agg("ratio")
        rows[arm] = dict(seeds=per, fwd_mean=f_m, fwd_sd=f_s, dM_mean=m_m, dM_sd=m_s,
                         ratio_mean=r_m, ratio_sd=r_s,
                         dT_mean=agg("dT_dlogKo")[0])
        print(f"{arm:>12} {f_m:8.4f}+-{f_s:.4f} {m_m:11.4f}+-{m_s:.4f} "
              f"{r_m*100:10.2f}+-{r_s*100:.2f}%")

    print("\n  |d<M>/dlogKo| is the severed pathway; |d<T>/dlogKo| is intact and shows the network")
    print("  did learn Ko. Their ratio is the leakage floor.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(setup=dict(n_coll=N_COLL, iters=ITERS, lbfgs=LBFGS_ITERS, width=WIDTH,
                                              depth=DEPTH, seed=SEED, nominal=NOMINAL),
                                   solver_reference=ref, arms=rows), indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
