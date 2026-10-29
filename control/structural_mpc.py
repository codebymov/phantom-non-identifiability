"""Closed-loop comparison of SIMULATOR STRUCTURE, not of controller brands.

The paper shows that dropping the Posnov coupling fabricates a null in the parameter-to-observation
Jacobian. control/input_jacobian_sweep.py shows the same reduction exactly zeros the
INPUT-to-moisture Jacobian. This script asks the decision-relevant question: does a controller built
on the reduced simulator make systematically worse decisions on a coupled truth plant?

Task (a genuine trade-off that exists only when Pn > 0). Dry a sheet to a terminal mean-moisture
target while holding the surface at a quality setpoint, without exceeding a scorching limit:

    min  q_T * sum_k (Tsurf_k - T_SET)^2  +  q_M * (Mbar_end - M_TARGET)^2
         + r * sum_k (du_k)^2  +  pen * sum_k relu(Tsurf_k - T_CRIT)^2

Raising the air temperature u raises Tsurf (helps the setpoint) but, through the thermogradient
term, drives moisture from the hot surface into the sheet and leaves it WETTER (hurts the moisture
target). That trade-off is real at Pn > 0 and *invisible* at Pn = 0, where dMbar/du == 0 exactly.

Controllers differ ONLY in the prediction model's Pn; horizon, cost, bounds, solver and tuning are
identical, so the independent variable is simulator structure:

  reduced    Pn_model = 0                    (the standard simplification)
  matched    Pn_model = Pn_true              (oracle)
  uncertain  spike-and-slab over Pn          (mass pi_0 at 0, rest on a positive support)

Idealisations, stated plainly: full-state feedback (the controller is handed the truth plant's T and
M fields; no observer), and a shrinking-horizon solver-in-the-loop NMPC rather than a learned
surrogate. Both are deliberate -- they remove estimator and surrogate quality as confounds so that
the only difference between arms is the structure of the prediction model.

Run:  PYTHONPATH=. .venv/bin/python control/structural_mpc.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from control.coupled_plant import make_plant

DTAU = 0.01
N_STEP = 200                 # plant steps over tau in (0, 2]
CTRL_EVERY = 10              # control decision every 10 plant steps (20 decisions)
U_MAX = 2.0
N_BLOCK = 3                  # piecewise-constant control blocks over the remaining horizon
N_STARTS = 4                 # multistart per solve, to rule out local-basin artefacts

T_SET, T_CRIT = 0.55, 0.65   # quality setpoint and scorching limit (surface temperature)
M_TARGET = 0.565             # terminal mean-moisture target
Q_T, Q_M, R_DU, PEN = 1.0, 400.0, 0.05, 200.0

PN_GRID_UNCERTAIN = [0.0, 0.2, 0.4, 0.6]
PI0 = 0.5                    # spike mass on Pn = 0; remainder split evenly over the positive support
OUT = Path(__file__).resolve().parents[1] / "runs/control/structural_mpc.json"


def _weights():
    w = np.full(len(PN_GRID_UNCERTAIN), (1.0 - PI0) / (len(PN_GRID_UNCERTAIN) - 1))
    w[PN_GRID_UNCERTAIN.index(0.0)] = PI0
    return w


def rollout_cost(pn_model, T0, M0, u_prev, plan, n_left):
    """Predicted cost-to-go under a model with coupling pn_model, from the given full state."""
    pl = make_plant(pn_model, dtau=DTAU)
    pl.T, pl.M = T0.copy(), M0.copy()
    edges = np.linspace(0, n_left, N_BLOCK + 1).astype(int)
    J, up = 0.0, u_prev
    Mbar = M0.mean()
    for b in range(N_BLOCK):
        u = float(plan[b])
        J += R_DU * (u - up) ** 2
        for _ in range(edges[b + 1] - edges[b]):
            s = pl.step(u)
            Tsurf, Mbar = s[4], s[0]
            J += Q_T * (Tsurf - T_SET) ** 2 * DTAU
            J += PEN * max(0.0, Tsurf - T_CRIT) ** 2 * DTAU
        up = u
    J += Q_M * (Mbar - M_TARGET) ** 2
    return J


def solve_move(kind, pn_true, T0, M0, u_prev, n_left):
    """One receding-horizon solve. `kind` selects the prediction model's structure."""
    if kind == "reduced":
        models, wts = [0.0], [1.0]
    elif kind == "matched":
        models, wts = [pn_true], [1.0]
    else:
        models, wts = PN_GRID_UNCERTAIN, _weights()

    def obj(plan):
        p = np.clip(plan, 0.0, U_MAX)
        return float(sum(w * rollout_cost(pm, T0, M0, u_prev, p, n_left)
                         for pm, w in zip(models, wts)))

    # Multistart. A receding-horizon solve from a single warm start can stop in a local basin, and
    # without this the model-uncertain arm's occasional edge over the *matched* arm cannot be
    # distinguished from an optimiser artefact -- a correct prediction model should not lose.
    starts = [np.full(N_BLOCK, float(np.clip(u_prev, 0.05, U_MAX - 0.05)))]
    rng = np.random.default_rng(N_BLOCK * 7 + n_left)
    starts += [rng.uniform(0.0, U_MAX, N_BLOCK) for _ in range(N_STARTS - 1)]
    best, best_J = None, np.inf
    for x0 in starts:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=[(0.0, U_MAX)] * N_BLOCK,
                       options=dict(maxiter=40, ftol=1e-10, eps=1e-4))
        if float(res.fun) < best_J:
            best_J, best = float(res.fun), res.x
    return float(np.clip(best[0], 0.0, U_MAX))


def run(kind, pn_true):
    """Closed loop: NMPC of the given structure driving the coupled truth plant."""
    truth = make_plant(pn_true, dtau=DTAU); truth.reset()
    u_prev, t0 = 0.5, time.time()
    log = {"u": [], "Mbar": [], "Tsurf": [], "tau": []}
    n_solves = 0
    for k in range(N_STEP):
        if k % CTRL_EVERY == 0:
            n_left = N_STEP - k
            u_prev = solve_move(kind, pn_true, truth.T, truth.M, u_prev, n_left)
            n_solves += 1
        s = truth.step(u_prev)
        log["u"].append(u_prev); log["Mbar"].append(s[0]); log["Tsurf"].append(s[4])
        log["tau"].append(truth.tau)
    wall = time.time() - t0
    log = {k: np.array(v) for k, v in log.items()}
    log["ms_per_solve"] = wall / max(n_solves, 1) * 1e3
    return log


def score(log):
    Ts, Mb, u = log["Tsurf"], log["Mbar"], log["u"]
    track = float(np.sum((Ts - T_SET) ** 2) * DTAU)
    mterm = float((Mb[-1] - M_TARGET) ** 2)
    viol = float(np.max(np.maximum(Ts - T_CRIT, 0.0)))
    effort = float(np.sum(np.diff(np.r_[u[0], u]) ** 2))
    total = Q_T * track + Q_M * mterm + R_DU * effort + PEN * float(
        np.sum(np.maximum(Ts - T_CRIT, 0.0) ** 2) * DTAU)
    return dict(J_true=total, temp_track=track, M_final=float(Mb[-1]),
                M_err=float(abs(Mb[-1] - M_TARGET)), T_viol=viol, effort=effort,
                u_mean=float(u.mean()), ms_per_solve=float(log["ms_per_solve"]))


def main() -> None:
    print(f"task: reach Mbar_end = {M_TARGET} and hold Tsurf at {T_SET}, with Tsurf <= {T_CRIT}")
    print(f"      control every {CTRL_EVERY} steps ({N_STEP // CTRL_EVERY} decisions), "
          f"{N_BLOCK} blocks, shrinking horizon, full-state feedback\n")
    results = {}
    for pn_true in [0.1, 0.2, 0.35, 0.5, 0.6, 0.8]:
        print(f"=== truth plant Pn = {pn_true} ===")
        print(f"  {'controller':>10} {'J_true':>9} {'M_final':>8} {'|M err|':>8} "
              f"{'T track':>8} {'T viol':>7} {'u mean':>7} {'ms/solve':>9}")
        results[str(pn_true)] = {}
        for kind in ("reduced", "matched", "uncertain"):
            log = run(kind, pn_true)
            m = score(log)
            results[str(pn_true)][kind] = dict(
                metrics=m, u=log["u"].tolist(), Mbar=log["Mbar"].tolist(),
                Tsurf=log["Tsurf"].tolist())
            print(f"  {kind:>10} {m['J_true']:9.4f} {m['M_final']:8.4f} {m['M_err']:8.4f} "
                  f"{m['temp_track']:8.4f} {m['T_viol']:7.4f} {m['u_mean']:7.3f} "
                  f"{m['ms_per_solve']:9.0f}")
        r = results[str(pn_true)]
        gap = (r["reduced"]["metrics"]["J_true"] - r["matched"]["metrics"]["J_true"])
        rel = gap / max(r["matched"]["metrics"]["J_true"], 1e-12) * 100
        print(f"  -> reduced-model regret vs matched: {gap:+.4f} ({rel:+.1f}%)\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        setup=dict(dtau=DTAU, n_step=N_STEP, ctrl_every=CTRL_EVERY, n_block=N_BLOCK,
                   T_SET=T_SET, T_CRIT=T_CRIT, M_TARGET=M_TARGET, U_MAX=U_MAX,
                   Q_T=Q_T, Q_M=Q_M, R_DU=R_DU, PEN=PEN,
                   pn_grid_uncertain=PN_GRID_UNCERTAIN, pi0=PI0),
        results=results, n_starts=N_STARTS), indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
