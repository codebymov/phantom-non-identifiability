"""Control-oriented PINC + NMPC on the REAL Luikov system (control/luikov_plant.py).

Faithful drying-process control problem: heat the web interior to a target temperature as fast as
possible via the hot-air temperature u, WITHOUT scorching the surface (T_surf <= T_crit). Note an
honest consequence of the simplified Luikov system: moisture evolves autonomously (independent of
temperature and of u), so air temperature controls TEMPERATURE, not drying rate -- the control
analogue of the paper's observability result. Controlling drying rate needs the T->M coupling the
simplified model drops (a stated limitation / motivation for the fuller model).

Pipeline: (1) roll the plant under random controls to build a physics-generated dataset;
(2) train a control-oriented surrogate f(s,u)->ds (the PINC, differentiable predictor);
(3) run gradient-based receding-horizon NMPC through it, vs a PID baseline;
(4) export the surrogate to ONNX for the MATLAB/Simulink path.

Run:  OMP_NUM_THREADS=4 .venv/bin/python control/pinc_nmpc.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from luikov_plant import LuikovPlant  # noqa: E402

from kibapap.physics.luikov import LuikovParams  # noqa: E402

RNG = np.random.default_rng(0)
DTAU = 0.01
U_MAX, DU_MAX = 2.0, 0.2          # air-temperature actuator range + slew
# Aggressive surface-temperature setpoint ABOVE the scorching limit: objective and constraint
# genuinely conflict, so the controller must trade tracking for safety.
T_SET = 0.70                     # surface-temperature setpoint (Sollwert), above the limit
T_CRIT = 0.55                    # surface-temperature safety limit (Beschraenkung)
NS = 6                           # reduced-state dimension
IMID, ISURF = 5, 4               # obs indices: interior temp, surface temp (the controlled var)


def plant():
    return LuikovPlant(LuikovParams(0.5, 2.0, 1.0, 1.0), n_x=101, dtau=DTAU)


# ------------------------------------------------------------------ dataset (physics rollouts) --
def make_data(n_ep=120, steps=180):
    Z, Y = [], []
    for _ in range(n_ep):
        pl = plant(); s = pl.reset()
        u = RNG.uniform(0, U_MAX)
        for _ in range(steps):
            u = float(np.clip(u + RNG.normal(0, 0.1), 0, U_MAX))   # smooth random control
            s2 = pl.step(u)
            Z.append(np.r_[s, u]); Y.append(s2 - s); s = s2
    return np.array(Z), np.array(Y)


class Surrogate(torch.nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(NS + 1, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(),
            torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, NS))
        for nm in ("xmu", "xsd", "ymu", "ysd"):
            self.register_buffer(nm, torch.zeros(NS + 1 if nm[0] == "x" else NS))

    def forward(self, z):
        return self.net((z - self.xmu) / self.xsd) * self.ysd + self.ymu


def train(Z, Y):
    m = Surrogate()
    with torch.no_grad():
        f32 = torch.float32
        m.xmu.copy_(torch.tensor(Z.mean(0), dtype=f32)); m.xsd.copy_(torch.tensor(Z.std(0) + 1e-8, dtype=f32))
        m.ymu.copy_(torch.tensor(Y.mean(0), dtype=f32)); m.ysd.copy_(torch.tensor(Y.std(0) + 1e-8, dtype=f32))
    Zt = torch.tensor(Z, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    opt = torch.optim.Adam(m.net.parameters(), lr=2e-3)
    for _ in range(4000):
        i = torch.randint(0, len(Zt), (4096,))
        opt.zero_grad(); (((m(Zt[i]) - Yt[i]) ** 2).mean()).backward(); opt.step()
    return m, float(((m(Zt) - Yt) ** 2).mean().detach())


# ------------------------------------------------------------------------------ controllers -----
def nmpc_move(m, s, u_prev, Np=25, Nc=5, q=1.0, r=1e-3, pen=200.0, iters=60):
    s0 = torch.tensor(s, dtype=torch.float32)
    theta = torch.full((Nc,), float(np.clip(u_prev / U_MAX, 1e-3, 1 - 1e-3)))
    theta = torch.logit(theta).clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.15)
    for _ in range(iters):
        opt.zero_grad(); s = s0; J = 0.0; up = torch.tensor(u_prev, dtype=torch.float32)
        plan = torch.sigmoid(theta) * U_MAX
        for k in range(Np):
            uk = plan[min(k, Nc - 1)]
            s = s + m(torch.cat([s, uk.reshape(1)]))
            J = J + q * (s[ISURF] - T_SET) ** 2 + r * (uk - up) ** 2
            J = J + pen * torch.relu(s[ISURF] - T_CRIT) ** 2
            up = uk
        J.backward(); opt.step()
    u0 = float(torch.sigmoid(theta[0]).detach()) * U_MAX
    return float(np.clip(u0, max(0.0, u_prev - DU_MAX), min(U_MAX, u_prev + DU_MAX)))


# PI control (industrial standard); Kd=0 because the surface responds fast at small dtau.
def pid(e, integ, e_prev, Kp=12.0, Ki=1.5, Kd=0.0):
    integ += e * DTAU
    u = Kp * e + Ki * integ + Kd * (e - e_prev) / DTAU
    return float(np.clip(u, 0, U_MAX)), integ


def run(kind, m=None, N=200):
    pl = plant(); s = pl.reset(); u_prev = 0.0; integ = 0.0; e_prev = 0.0
    log = {"Tmid": [], "Tsurf": [], "Mbar": [], "u": []}
    for _ in range(N):
        if kind == "pid":
            e = T_SET - s[ISURF]; u, integ = pid(e, integ, e_prev); e_prev = e
            u = float(np.clip(u, u_prev - DU_MAX, u_prev + DU_MAX))
        else:
            u = nmpc_move(m, s, u_prev)
        s = pl.step(u); u_prev = u
        for key, v in zip(["Tmid", "Tsurf", "Mbar", "u"], [s[IMID], s[ISURF], s[0], u]):
            log[key].append(v)
    return {k: np.array(v) for k, v in log.items()}


def metrics(log):
    Ts = log["Tsurf"]
    viol = float(np.max(np.maximum(Ts - T_CRIT, 0)))          # scorching (over the limit)
    return dict(tcrit_viol=viol, surf_final=float(Ts[-1]), surf_max=float(Ts.max()))


def export_onnx(m):
    m.eval()
    dummy = torch.zeros(1, NS + 1)          # input = [state(6), u] ; output = dstate(6)
    path = HERE / "pinc_predictor.onnx"
    torch.onnx.export(m, dummy, str(path), input_names=["state_u"], output_names=["dstate"],
                      opset_version=17)
    return path


def main():
    print("generating physics rollouts + training PINC...")
    Z, Y = make_data(); m, mse = train(Z, Y)
    print(f"  PINC 1-step val MSE = {mse:.2e}  (states: Mbar,M0,Mmid,Tbar,Tsurf,Tmid)\n")
    out = {}
    for name, kind, mm in [("PID", "pid", None), ("PINC-NMPC", "nmpc", m)]:
        t0 = time.time(); log = run(kind, mm); wall = time.time() - t0
        mt = metrics(log); out[name] = log
        print(f"{name:10s} surf max {mt['surf_max']:.3f} final {mt['surf_final']:.3f}  "
              f"scorch-viol {mt['tcrit_viol']:.3f} (limit {T_CRIT})  "
              f"({wall:.1f}s, {wall/len(log['u'])*1000:.0f}ms/step)")
    try:
        path = export_onnx(m); print(f"\nexported PINC -> {path} (for MATLAB importNetworkFromONNX)")
    except Exception as e:
        print(f"\n[ONNX export skipped: {e}]")
    np.savez_compressed(HERE / "pinc_result.npz",
                        **{f"{n}_{k}": v for n, log in out.items() for k, v in log.items()},
                        dtau=DTAU, t_set=T_SET, t_crit=T_CRIT)
    print(f"saved {HERE / 'pinc_result.npz'}")


if __name__ == "__main__":
    main()
