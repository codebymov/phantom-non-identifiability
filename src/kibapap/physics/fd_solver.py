"""Implicit-Euler finite-difference solver for the 1D dimensionless Luikov system.

Strategy: operator-split coupling between the moisture and thermal equations. The
moisture PDE is decoupled from T (we have already dropped the thermogradient cross
term in the moisture balance), so the time step proceeds in two tridiagonal solves:

    1) Solve M^{n+1} from M^n via implicit Euler:
       (I - dtau * Lu * D2_M) M^{n+1} = M^n + dtau * Lu * g_M
    2) Compute dM/dtau approx (M^{n+1} - M^n) / dtau
    3) Solve T^{n+1} from T^n via implicit Euler with M-source:
       (I - dtau * D2_T) T^{n+1} = T^n + dtau * Ko * dM/dtau + dtau * g_T

D2_T and D2_M are 1D second-derivative operators with Robin BC at xi=0 absorbed into
the matrix and a constant g_BC source vector. Robin coefficients are Bi_t / Bi_m. The
back boundary at xi=1 is Neumann (zero flux).

The resulting tridiagonal system is solved with scipy.linalg.solve_banded; the cost is
O(N) per step, total O(N * n_t). For typical problem sizes (N=101, n_t=200) the solver
runs in well under a second on a laptop CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from kibapap.physics.luikov import LuikovParams


@dataclass
class FDSolution:
    """Container for FD solver output."""

    xi: np.ndarray  # (n_x,)
    tau: np.ndarray  # (n_t,)
    T: np.ndarray  # (n_t, n_x), dimensionless temperature field
    M: np.ndarray  # (n_t, n_x), dimensionless moisture field
    params: LuikovParams

    @property
    def shape(self) -> tuple[int, int]:
        return self.T.shape


def _build_d2_operator(
    n_x: int, h: float, bi_left: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sub/diag/sup bands plus a constant BC source vector.

    The discretization is centered second-order finite differences with ghost-point
    elimination at xi=0 (Robin) and xi=1 (Neumann).

    For Robin BC dT/dxi|_0 = bi * (T_0 - T_inf), with T_inf passed via the source,
    the row 0 contributes to the matrix:

        diag[0]   = -2 (1 + h * bi) / h^2
        super[0]  =  2 / h^2

    and to the constant source vector g[0] = 2 * bi * T_inf / h. T_inf is set by the
    caller via a separate scalar (g is returned with T_inf factored as 1, so the caller
    multiplies by the correct boundary value, 1.0 for T or 0.0 for M).

    Parameters
    ----------
    n_x : int
        Number of spatial nodes including both boundaries.
    h : float
        Grid spacing, h = 1 / (n_x - 1).
    bi_left : float
        Biot number at xi=0.

    Returns
    -------
    sub, diag, sup : np.ndarray of shape (n_x,)
        Tridiagonal bands of the second-derivative operator. sub[0] and sup[-1] are
        unused (set to zero); they exist to match the band format of solve_banded.
    g : np.ndarray of shape (n_x,)
        Constant source vector with g[0] = 2 * bi_left / h, all other entries zero.
        The caller multiplies by T_inf (the boundary's external "drive" value).
    """
    sub = np.zeros(n_x)
    diag = np.zeros(n_x)
    sup = np.zeros(n_x)
    g = np.zeros(n_x)

    # Interior nodes
    for i in range(1, n_x - 1):
        sub[i] = 1.0 / h**2
        diag[i] = -2.0 / h**2
        sup[i] = 1.0 / h**2

    # Left (xi=0): Robin
    diag[0] = -2.0 * (1.0 + h * bi_left) / h**2
    sup[0] = 2.0 / h**2
    g[0] = 2.0 * bi_left / h

    # Right (xi=1): Neumann (insulated)
    diag[n_x - 1] = -2.0 / h**2
    sub[n_x - 1] = 2.0 / h**2

    return sub, diag, sup, g


def _to_banded(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray) -> np.ndarray:
    """Pack tridiagonal bands into the (3, n) layout that solve_banded expects.

    For solve_banded((1, 1), ab, b):
        ab[0, 1:]   = super-diagonal entries (ab[0, 0] unused)
        ab[1, :]    = main diagonal
        ab[2, :-1]  = sub-diagonal entries (ab[2, -1] unused)
    """
    n = diag.shape[0]
    ab = np.zeros((3, n))
    ab[0, 1:] = sup[:-1]
    ab[1, :] = diag
    ab[2, :-1] = sub[1:]
    return ab


def solve_luikov_fd(
    params: LuikovParams,
    n_x: int = 101,
    n_t: int = 201,
    tau_final: float = 2.0,
) -> FDSolution:
    """Solve the 1D dimensionless Luikov system on a uniform grid.

    Parameters
    ----------
    params : LuikovParams
        Lu, Ko, Bi_t, Bi_m.
    n_x : int
        Number of spatial nodes (including boundaries). 101 is a sensible default.
    n_t : int
        Number of time levels (including t=0). 201 gives dtau ~= 0.01 for tau_final=2.
    tau_final : float
        Final dimensionless time. tau_final ~ 2 is enough for full equilibration at
        the baseline parameter set; raise if Lu is small.

    Returns
    -------
    FDSolution dataclass with xi, tau, T (n_t, n_x), M (n_t, n_x), params.
    """
    h = 1.0 / (n_x - 1)
    dtau = tau_final / (n_t - 1)
    xi = np.linspace(0.0, 1.0, n_x)
    tau = np.linspace(0.0, tau_final, n_t)

    # Build operators
    sub_T, diag_T, sup_T, g_T_unit = _build_d2_operator(n_x, h, params.Bi_t)
    sub_M, diag_M, sup_M, g_M_unit = _build_d2_operator(n_x, h, params.Bi_m)

    # The thermal BC drives toward T_inf = 1; the moisture BC toward M_inf = 0.
    g_T = g_T_unit * 1.0
    g_M = g_M_unit * 0.0  # M_inf_dimless = 0 since we scaled by (M0 - Meq).

    # Apply the Lu factor to the moisture diffusion operator and its BC source.
    sub_M *= params.Lu
    diag_M *= params.Lu
    sup_M *= params.Lu
    g_M *= params.Lu

    # Implicit-Euler matrices: I - dtau * L
    ab_T = _to_banded(-dtau * sub_T, 1.0 - dtau * diag_T, -dtau * sup_T)
    ab_M = _to_banded(-dtau * sub_M, 1.0 - dtau * diag_M, -dtau * sup_M)

    # Storage
    T = np.zeros((n_t, n_x))
    M = np.zeros((n_t, n_x))
    T[0, :] = 0.0
    M[0, :] = 1.0

    for n in range(n_t - 1):
        # 1) Moisture step.  With Pn > 0 the moisture balance carries the thermogradient
        #    (Soret) cross-term Lu*Pn*d2T/dxi2, evaluated at the current (lagged) T^n so the
        #    operator split stays two tridiagonal solves (explicit coupling, O(dtau), the
        #    same order as the implicit-Euler time discretisation). d2T is the SAME discrete
        #    Laplacian used by the thermal operator, including the Robin-BC drive g_T.
        rhs_M = M[n] + dtau * g_M
        if params.Pn != 0.0:
            Tn = T[n]
            d2T = diag_T * Tn + g_T
            d2T[1:] += sub_T[1:] * Tn[:-1]
            d2T[:-1] += sup_T[:-1] * Tn[1:]
            rhs_M = rhs_M + dtau * params.Lu * params.Pn * d2T
        M[n + 1] = solve_banded((1, 1), ab_M, rhs_M)

        # 2) dM/dtau at the new step
        dMdtau = (M[n + 1] - M[n]) / dtau

        # 3) Thermal step with phase-change source
        rhs_T = T[n] + dtau * g_T + dtau * params.Ko * dMdtau
        T[n + 1] = solve_banded((1, 1), ab_T, rhs_T)

    return FDSolution(xi=xi, tau=tau, T=T, M=M, params=params)
