"""Control-oriented Luikov plant: the real 1D dimensionless drying PDE with a TIME-VARYING
hot-air temperature u(tau) as the manipulated variable (Stellgroesse).

Reuses the paper's benchmark FD operators (kibapap.physics.fd_solver) without modifying them, but
steps with a controllable thermal Robin boundary: dT/dxi|_0 = Bi_t (T - u(tau)). This is the
"true system" (Regelstrecke) for the drying controller. Higher air temperature u dries faster but
raises the surface temperature T(0,tau); the control must respect a scorching limit T_surf <= T_crit.

Reduced observation for control: [ mean moisture Mbar, surface moisture M0, mean temp Tbar,
surface temp T0 ] -- Mbar is the drying progress, T0 the safety-constrained quantity.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from kibapap.physics.fd_solver import _build_d2_operator, _to_banded
from kibapap.physics.luikov import LuikovParams


class LuikovPlant:
    def __init__(self, params: LuikovParams = LuikovParams(), n_x: int = 101, dtau: float = 0.01):
        self.p = params; self.n_x = n_x; self.dtau = dtau
        h = 1.0 / (n_x - 1)
        sub_T, diag_T, sup_T, gT = _build_d2_operator(n_x, h, params.Bi_t)
        sub_M, diag_M, sup_M, gM = _build_d2_operator(n_x, h, params.Bi_m)
        sub_M *= params.Lu; diag_M *= params.Lu; sup_M *= params.Lu
        self.ab_T = _to_banded(-dtau * sub_T, 1.0 - dtau * diag_T, -dtau * sup_T)
        self.ab_M = _to_banded(-dtau * sub_M, 1.0 - dtau * diag_M, -dtau * sup_M)
        self.gT_unit = gT           # thermal boundary source for air temp = 1 -> scale by u
        self.gM = gM * 0.0          # moisture equilibrium is 0
        self.reset()

    def reset(self):
        self.T = np.zeros(self.n_x); self.M = np.ones(self.n_x); self.tau = 0.0
        return self.obs()

    def step(self, u: float):
        """Advance one dtau with hot-air temperature u (normalized, ~[0,1])."""
        rhs_M = self.M + self.dtau * self.gM
        M_new = solve_banded((1, 1), self.ab_M, rhs_M)
        dMdt = (M_new - self.M) / self.dtau
        rhs_T = self.T + self.dtau * self.gT_unit * float(u) + self.dtau * self.p.Ko * dMdt
        T_new = solve_banded((1, 1), self.ab_T, rhs_T)
        self.M, self.T = M_new, T_new; self.tau += self.dtau
        return self.obs()

    def obs(self) -> np.ndarray:
        mid = self.n_x // 2
        return np.array([self.M.mean(), self.M[0], self.M[mid],
                         self.T.mean(), self.T[0], self.T[mid]])


def _selftest():
    """At constant u=1 the plant must reproduce the benchmark FD solver."""
    from kibapap.physics import solve_luikov_fd
    p = LuikovParams(0.5, 2.0, 1.0, 1.0)
    ref = solve_luikov_fd(p, n_x=101, n_t=201, tau_final=2.0)
    pl = LuikovPlant(p, n_x=101, dtau=2.0 / 200); pl.reset()
    for _ in range(200):
        pl.step(1.0)
    dM = abs(pl.M.mean() - ref.M[-1].mean()); dT = abs(pl.T[0] - ref.T[-1, 0])
    print(f"selftest vs benchmark FD @ tau=2: |dMbar|={dM:.2e}  |dT_surf|={dT:.2e}")
    assert dM < 1e-6 and dT < 1e-6, "control plant diverges from benchmark solver"
    print("OK: control plant matches the benchmark solver at constant u=1")


if __name__ == "__main__":
    _selftest()
