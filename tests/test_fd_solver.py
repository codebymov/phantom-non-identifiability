"""Smoke tests for the FD Luikov solver.

Run with: pytest tests/

These check the physics is right: monotonicity, asymptotic limits, mass loss positivity.
They do not test exact numerical accuracy (that's the FD's job).
"""

from __future__ import annotations

import numpy as np

from kibapap.physics import LuikovParams, solve_luikov_fd


def test_initial_conditions():
    """At tau=0, T should be 0 everywhere and M should be 1 everywhere."""
    sol = solve_luikov_fd(LuikovParams(), n_x=51, n_t=51, tau_final=1.0)
    assert np.allclose(sol.T[0], 0.0)
    assert np.allclose(sol.M[0], 1.0)


def test_monotonicity_in_xi():
    """Once the boundary layer has spread, T is monotone non-increasing in xi (drier
    surface is hotter) and M is monotone non-decreasing in xi (drier surface is drier).
    """
    sol = solve_luikov_fd(LuikovParams(), n_x=51, n_t=101, tau_final=2.0)
    # Skip the initial transient where the wet-bulb cooling can cause non-monotonicity in T
    for n in range(50, 101):
        # T monotone non-increasing
        assert np.all(np.diff(sol.T[n]) <= 1e-10), f"T non-monotone at n={n}"
        # M monotone non-decreasing
        assert np.all(np.diff(sol.M[n]) >= -1e-10), f"M non-monotone at n={n}"


def test_mass_only_decreases():
    """Average dimensionless moisture can only decrease over time (no remoistening
    in this BC setup).
    """
    sol = solve_luikov_fd(LuikovParams(), n_x=51, n_t=101, tau_final=2.0)
    avg_M = sol.M.mean(axis=1)
    diffs = np.diff(avg_M)
    assert np.all(diffs <= 1e-12), "Average M increased somewhere"


def test_asymptotic_steady_state():
    """For long enough tau, T should approach 1 and M should approach 0 (steady state)."""
    sol = solve_luikov_fd(LuikovParams(Lu=1.0, Ko=0.5), n_x=51, n_t=501, tau_final=20.0)
    # Steady-state values
    assert sol.T[-1].mean() > 0.95, f"T did not approach 1: {sol.T[-1].mean()}"
    assert sol.M[-1].mean() < 0.05, f"M did not approach 0: {sol.M[-1].mean()}"


def test_higher_Bi_t_gives_faster_heating():
    """Higher thermal Biot should heat the surface faster at any fixed tau."""
    sol_low = solve_luikov_fd(LuikovParams(Bi_t=0.5), n_x=51, n_t=101, tau_final=1.0)
    sol_high = solve_luikov_fd(LuikovParams(Bi_t=3.0), n_x=51, n_t=101, tau_final=1.0)
    # Compare at tau=0.5
    idx = 50
    assert sol_high.T[idx, 0] > sol_low.T[idx, 0]


def test_higher_Lu_gives_faster_drying():
    """Higher Luikov number means faster moisture diffusion, so dryer at fixed tau."""
    sol_low = solve_luikov_fd(LuikovParams(Lu=0.2), n_x=51, n_t=101, tau_final=1.0)
    sol_high = solve_luikov_fd(LuikovParams(Lu=1.0), n_x=51, n_t=101, tau_final=1.0)
    idx = 50
    assert sol_high.M[idx].mean() < sol_low.M[idx].mean()


def test_grid_independence_loose():
    """Solutions on coarser and finer grids should be close (not exact)."""
    sol_coarse = solve_luikov_fd(LuikovParams(), n_x=51, n_t=101, tau_final=2.0)
    sol_fine = solve_luikov_fd(LuikovParams(), n_x=201, n_t=401, tau_final=2.0)
    # Compare surface T at the final time
    diff = abs(sol_coarse.T[-1, 0] - sol_fine.T[-1, 0])
    assert diff < 0.02, f"Grid-coarse vs fine surface T differ by {diff}"
