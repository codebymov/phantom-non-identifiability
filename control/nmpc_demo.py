"""Differentiable-surrogate NMPC proof-of-concept for hot-air web drying.

Concrete realization of the paper's amortization niche (Sec. 5.7): a solver-in-the-loop predictor
is far too slow for a real-time control loop, but a fast DIFFERENTIABLE learned surrogate enables
gradient-based receding-horizon control. This PoC closes the loop on a lumped drying plant and
compares a surrogate-NMPC against a classical PID baseline (the Regelungstechnik reference).

Framing (DIN 19226 / Regelkreis):
  Regelstrecke (plant)      : lumped drying dynamics (material temperature T_G, moisture X)
  Regelgroesse y (Istwert)  : T_G
  Fuehrungsgroesse w (Soll) : T_set
  Regeldifferenz e = w - y
  Stellgroesse u            : heating power Q_heat, with magnitude + slew-rate limits
  Stoergroesse d            : incoming web moisture / ambient temperature step
  Zustandsbeschraenkung     : T_G <= T_critical (avoid scorching the web)

The surrogate stands in for a control-oriented PINN (PINC); the point is the *architecture* and the
differentiability that makes gradient-based NMPC real-time. Plant is a demonstration model, not the
full Luikov PDE -- see README for the path to the real predictor.

Run:  .venv/bin/python control/nmpc_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

RNG = np.random.default_rng(0)
DT = 10.0                      # control step (s); thermal drying is slow
# --- plant (lumped) parameters ---
M_CP = 4000.0                  # thermal mass * cp  (J/K)
DHV = 2.4e6                    # latent heat of vaporization (J/kg)
K_EVAP = 8e-7                  # evaporation-rate coefficient
Q_MAX, DQ_MAX = 6000.0, 800.0  # actuator magnitude / slew limits (W, W per step)
T_CRIT = 72.0                  # state constraint (deg C) -- binds on an aggressive step to 70
T_AMB0 = 20.0


def m_vap(Tg, X):
    """Nonlinear evaporation rate: moisture x superheat driving force (kg/s)."""
    return K_EVAP * np.maximum(X, 0.0) * np.maximum(Tg - T_AMB0, 0.0)


def plant_step(state, u, d_moist=0.0):
    """One RK4 step of the drying plant. state=(T_G, X); u=Q_heat; d_moist adds incoming moisture."""
    def f(s):
        Tg, X = s
        mv = m_vap(Tg, X)
        return np.array([(u - mv * DHV) / M_CP, -mv + d_moist])
    s = np.asarray(state, float)
    k1 = f(s); k2 = f(s + 0.5 * DT * k1); k3 = f(s + 0.5 * DT * k2); k4 = f(s + DT * k3)
    return s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------- surrogate (control-oriented) --
class Surrogate(torch.nn.Module):
    """(T_G, X, Q) -> (dT_G, dX) one-step dynamics; differentiable predictor for the NMPC."""
    def __init__(self, h=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(),
            torch.nn.Linear(h, 2))
        self.xmu = self.xsd = self.ymu = self.ysd = None

    def forward(self, z):
        return self.net((z - self.xmu) / self.xsd) * self.ysd + self.ymu


def train_surrogate():
    # random-control rollouts of the plant for training data
    Z, Y = [], []
    for _ in range(400):
        s = np.array([RNG.uniform(20, 70), RNG.uniform(0.02, 0.5)])
        for _ in range(40):
            u = RNG.uniform(0, Q_MAX)
            s2 = plant_step(s, u)
            Z.append([s[0], s[1], u]); Y.append(s2 - s)
            s = s2
            if s[0] > 90 or s[1] <= 0:
                break
    Z = np.array(Z); Y = np.array(Y)
    m = Surrogate()
    m.xmu = torch.tensor(Z.mean(0), dtype=torch.float32); m.xsd = torch.tensor(Z.std(0) + 1e-6, dtype=torch.float32)
    m.ymu = torch.tensor(Y.mean(0), dtype=torch.float32); m.ysd = torch.tensor(Y.std(0) + 1e-6, dtype=torch.float32)
    Zt = torch.tensor(Z, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    opt = torch.optim.Adam(m.net.parameters(), lr=2e-3)
    for _ in range(3000):
        i = torch.randint(0, len(Zt), (2048,))
        opt.zero_grad(); (((m(Zt[i]) - (Yt[i])) ** 2).mean()).backward(); opt.step()
    val = float(((m(Zt) - Yt) ** 2).mean().detach())
    return m, val


# --------------------------------------------------------------------------- surrogate NMPC ------
def nmpc_move(m, state, u_prev, w_seq, Np, Nc, q=1.0, r=2e-6, iters=60):
    """Gradient-based receding-horizon step through the differentiable surrogate.
    Optimizes an Nc-move control plan (held then frozen over Np), returns the first move."""
    s0 = torch.tensor(state, dtype=torch.float32)
    # decision variables in [0,1]; map to [0,Qmax] via sigmoid (magnitude constraint by construction)
    theta = torch.zeros(Nc, requires_grad=True)
    with torch.no_grad():
        theta += torch.logit(torch.tensor(np.clip(u_prev / Q_MAX, 1e-3, 1 - 1e-3), dtype=torch.float32))
    opt = torch.optim.Adam([theta], lr=0.15)
    w = torch.tensor(w_seq, dtype=torch.float32)
    for _ in range(iters):
        opt.zero_grad()
        u_plan = torch.sigmoid(theta) * Q_MAX
        s = s0; J = 0.0; u_prevk = torch.tensor(u_prev, dtype=torch.float32)
        for k in range(Np):
            uk = u_plan[min(k, Nc - 1)]
            ds = m(torch.cat([s, uk.reshape(1)]))
            s = s + ds
            J = J + q * (s[0] - w[min(k, len(w) - 1)]) ** 2
            J = J + r * (uk - u_prevk) ** 2
            J = J + 50.0 * torch.relu(s[0] - T_CRIT) ** 2                    # soft state constraint
            J = J + 5e-4 * torch.relu(torch.abs(uk - u_prevk) - DQ_MAX) ** 2  # soft slew
            u_prevk = uk
        J.backward(); opt.step()
    u0 = float(torch.sigmoid(theta[0]).detach()) * Q_MAX
    # hard slew + magnitude projection on the applied move
    u0 = float(np.clip(u0, u_prev - DQ_MAX, u_prev + DQ_MAX))
    return float(np.clip(u0, 0, Q_MAX))


def pid(e, integ, e_prev, Kp=420.0, Ki=9.0, Kd=70.0):
    integ = integ + e * DT
    u = Kp * e + Ki * integ + Kd * (e - e_prev) / DT
    return float(np.clip(u, 0, Q_MAX)), integ


def run(controller, m=None, T=120):
    """Closed-loop simulation. Setpoint step at t=0->40 min region; moisture disturbance mid-run."""
    tgrid = np.arange(T)
    w = np.where(tgrid < 30, 45.0, 75.0)   # aggressive Sollwert 75 C ABOVE the 72 C safety limit
    s = np.array([T_AMB0 + 5.0, 0.45]); u_prev = 0.0; integ = 0.0; e_prev = 0.0
    log = {"T": [], "X": [], "u": [], "w": []}
    for k in range(T):
        d_moist = 4e-4 if 60 <= k < 66 else 0.0                # Stoergroesse: incoming wet web
        if controller == "pid":
            e = w[k] - s[0]; u, integ = pid(e, integ, e_prev); e_prev = e
        else:
            wseq = w[k:k + 30] if k + 30 <= T else np.r_[w[k:], np.full(30 - (T - k), w[-1])]
            u = nmpc_move(m, s, u_prev, wseq, Np=30, Nc=5)
        s = plant_step(s, u, d_moist); u_prev = u
        for key, v in zip(["T", "X", "u", "w"], [s[0], s[1], u, w[k]]):
            log[key].append(v)
    return {k: np.array(v) for k, v in log.items()}, tgrid


def metrics(log, w):
    T = log["T"]; e = w - T
    step_i = 30
    ss_err = float(np.mean(np.abs(e[-15:])))                    # bleibende Regelabweichung
    overshoot = float(max(0.0, (T[step_i:].max() - w[-1]) / w[-1] * 100))
    settled = np.where(np.abs(e[step_i:]) < 0.02 * w[-1])[0]
    settle = float(settled[0] * DT / 60) if len(settled) else np.nan  # min
    iae = float(np.sum(np.abs(e)) * DT)
    viol = float(np.max(np.maximum(T - T_CRIT, 0)))
    return dict(ss_err=ss_err, overshoot=overshoot, settle_min=settle, iae=iae, tcrit_viol=viol)


def main():
    print("training control-oriented surrogate...")
    m, val = train_surrogate(); print(f"  surrogate 1-step val MSE = {val:.3e}\n")
    out = {}
    for name, ctrl, mm in [("PID", "pid", None), ("surrogate-NMPC", "nmpc", m)]:
        import time; t0 = time.time()
        log, tg = run(ctrl, mm); wall = time.time() - t0
        mt = metrics(log, log["w"])
        out[name] = (log, tg)
        print(f"{name:15s} ss-err {mt['ss_err']:.2f}C  overshoot {mt['overshoot']:.1f}%  "
              f"settle {mt['settle_min']:.1f}min  IAE {mt['iae']:.0f}  Tcrit-viol {mt['tcrit_viol']:.2f}C  "
              f"({wall:.1f}s, {wall/120*1000:.0f}ms/step)")
    OUT = Path(__file__).resolve().parent / "nmpc_result.npz"
    np.savez_compressed(OUT, **{f"{n}_{k}": v for n, (log, _) in out.items() for k, v in log.items()},
                        tgrid=out["PID"][1])
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
