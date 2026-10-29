"""Control plant with the thermogradient (Posnov) coupling restored.

`control/luikov_plant.py` implements the drying plant with a manipulated hot-air temperature u,
but its moisture update

    rhs_M = M + dtau * gM ;  M_new = solve_banded(ab_M, rhs_M)

references neither u nor T. That is the Pn = 0 reduction: the moisture field is autonomous, so
dM/du == 0 exactly and the thermal actuator has no authority over drying. (The module docstring
nonetheless states "Higher air temperature u dries faster", which the implementation cannot deliver
-- the control-side instance of the same phantom this paper is about.)

This subclass restores the coupling without modifying the original file. The moisture balance gains
the lagged-explicit thermogradient term used by the benchmark solver
(`kibapap.physics.fd_solver`, same operator and same O(dtau) split):

    dM/dtau = Lu d2M/dxi2 + Lu Pn d2T/dxi2 .

The discrete Laplacian d2T carries the thermal Robin drive, which here is the *manipulated* air
temperature u rather than a fixed 1. So with Pn > 0 the input reaches the moisture field by two
routes: directly through that boundary drive, and indirectly through the evolving T field. Both are
O(Pn), and both vanish identically at Pn = 0.

At Pn = 0 this class is bit-identical to LuikovPlant (asserted in `_selftest`).

Run:  PYTHONPATH=. .venv/bin/python control/coupled_plant.py
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from kibapap.physics.fd_solver import _build_d2_operator
from kibapap.physics.luikov import LuikovParams

from control.luikov_plant import LuikovPlant


class CoupledLuikovPlant(LuikovPlant):
    """LuikovPlant plus the Lu*Pn*d2T/dxi2 term in the moisture balance."""

    def __init__(self, params: LuikovParams = LuikovParams(), n_x: int = 101, dtau: float = 0.01):
        super().__init__(params, n_x, dtau)
        h = 1.0 / (n_x - 1)
        # the same thermal second-derivative bands the parent used to build ab_T
        self.sub_T, self.diag_T, self.sup_T, _ = _build_d2_operator(n_x, h, params.Bi_t)
        self.Pn = float(getattr(params, "Pn", 0.0))

    def _d2T(self, u: float) -> np.ndarray:
        """Discrete d2T/dxi2 including the Robin drive toward the manipulated air temperature u."""
        T = self.T
        d2 = self.diag_T * T + self.gT_unit * float(u)
        d2[1:] += self.sub_T[1:] * T[:-1]
        d2[:-1] += self.sup_T[:-1] * T[1:]
        return d2

    def step(self, u: float):
        rhs_M = self.M + self.dtau * self.gM
        if self.Pn != 0.0:
            rhs_M = rhs_M + self.dtau * self.p.Lu * self.Pn * self._d2T(u)
        M_new = solve_banded((1, 1), self.ab_M, rhs_M)
        dMdt = (M_new - self.M) / self.dtau
        rhs_T = self.T + self.dtau * self.gT_unit * float(u) + self.dtau * self.p.Ko * dMdt
        T_new = solve_banded((1, 1), self.ab_T, rhs_T)
        self.M, self.T = M_new, T_new
        self.tau += self.dtau
        return self.obs()


def make_plant(pn: float, lu=0.5, ko=2.0, bit=1.0, bim=1.0, n_x=101, dtau=0.01):
    return CoupledLuikovPlant(LuikovParams(lu, ko, bit, bim, Pn=pn), n_x=n_x, dtau=dtau)


def _selftest():
    rng = np.random.default_rng(0)
    us = rng.uniform(0.0, 2.0, 150)

    ref, new = LuikovPlant(LuikovParams(0.5, 2.0, 1.0, 1.0)), make_plant(0.0)
    ref.reset(); new.reset()
    for u in us:
        a, b = ref.step(float(u)), new.step(float(u))
        assert np.allclose(a, b, rtol=0, atol=0), (a, b)
    print("bit-identity at Pn=0 vs LuikovPlant: OK (150 random control steps)")

    # the same coupling, run open-loop at constant u=1, must match the benchmark PDE solver
    from kibapap.physics import solve_luikov_fd
    for pn in (0.0, 0.3, 0.6):
        pl = make_plant(pn, dtau=2.0 / 200); pl.reset()
        for _ in range(200):
            pl.step(1.0)
        ref = solve_luikov_fd(LuikovParams(0.5, 2.0, 1.0, 1.0, Pn=pn), n_x=101, n_t=201,
                              tau_final=2.0)
        eM = np.linalg.norm(pl.M - ref.M[-1]) / np.linalg.norm(ref.M[-1])
        eT = np.linalg.norm(pl.T - ref.T[-1]) / np.linalg.norm(ref.T[-1])
        print(f"  Pn={pn:.1f}  vs benchmark solver at tau=2:  rel-L2 M {eM:.2e}   T {eT:.2e}")
        assert eM < 1e-12 and eT < 1e-12, (pn, eM, eT)
    print("agreement with kibapap.physics.solve_luikov_fd at u=1: OK")


if __name__ == "__main__":
    _selftest()
