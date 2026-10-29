"""Metamorphic layer: relations that must hold between runs, not values of single runs.

Scientific code has an oracle problem -- for most inputs nobody knows the right answer, so
asserting exact outputs only works where a closed form or a published number exists. What can
always be asserted is a *relation* between two runs: change an input in a known way, and the
output must change in a known way, or not at all.

That makes this the layer that can grow cheaply, and the one that catches the faults exact-value
tests miss. The first relation below is the paper's thesis stated as a metamorphic property:
at Pn = 0, perturbing Ko or Bi_t must leave the moisture field bit-for-bit unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from kibapap.physics import LuikovParams, solve_luikov_fd
from tests.paper_values import NOMINAL_THETA
from theory.observe import K_DEFAULT, channels, fisher, sample_times, solve

LU, KO, BIT, BIM = 0, 1, 2, 3


def _perturb(theta, index, factor):
    t = list(theta)
    t[index] *= factor
    return tuple(t)


@pytest.mark.parametrize("index", [KO, BIT])
def test_at_pn_zero_moisture_is_exactly_independent_of_the_temperature_parameters(index):
    """MR1, and the paper's central claim: not approximately independent, bit-for-bit identical.

    If this ever fails, either the reduction is no longer severing the path or the solver has
    developed a spurious coupling -- and the phantom-null result would be wrong either way.
    """
    base = solve(NOMINAL_THETA, 0.0)
    moved = solve(_perturb(NOMINAL_THETA, index, 1.5), 0.0)
    assert np.array_equal(base.M, moved.M), "moisture field must not move at all at Pn=0"
    assert not np.array_equal(base.T, moved.T), "the temperature field must move"


@pytest.mark.parametrize("index", [KO, BIT])
def test_restoring_the_coupling_restores_the_dependence(index):
    """MR2, the converse: at Pn > 0 the same perturbation must reach the moisture field."""
    base = solve(NOMINAL_THETA, 0.6)
    moved = solve(_perturb(NOMINAL_THETA, index, 1.5), 0.6)
    rel = np.linalg.norm(moved.M - base.M) / np.linalg.norm(base.M)
    assert rel > 1e-6, f"moisture must respond once the path exists (got {rel:.2e})"


@pytest.mark.parametrize("index", [LU, BIM])
def test_the_identifiable_parameters_always_reach_moisture(index):
    """MR3: Lu and Bi_m act on the moisture balance directly, at any coupling."""
    for pn in (0.0, 0.6):
        base = solve(NOMINAL_THETA, pn)
        moved = solve(_perturb(NOMINAL_THETA, index, 1.2), pn)
        rel = np.linalg.norm(moved.M - base.M) / np.linalg.norm(base.M)
        assert rel > 1e-6, (index, pn)


def test_the_bound_scales_linearly_with_the_noise_level():
    """MR4: F = S^T Sigma^-1 S, so scaling every sigma by k scales every CR bound by exactly k.

    Checked against a hand-built Fisher rather than through fisher(), because the noise fraction
    is a module constant bound at definition time and cannot be overridden by reassigning it.
    """
    sol = solve(NOMINAL_THETA, 0.6)
    ts = sample_times(sol, K_DEFAULT)
    d = 1e-3
    cols = []
    for j in range(4):
        up, dn = list(NOMINAL_THETA), list(NOMINAL_THETA)
        up[j] *= np.exp(d); dn[j] *= np.exp(-d)
        cu = channels(solve(tuple(up), 0.6), ts)
        cd = channels(solve(tuple(dn), 0.6), ts)
        cols.append(np.concatenate([(a - b) / (2 * d) for a, b in zip(cu, cd)]))
    S = np.column_stack(cols)
    base = channels(sol, ts)

    def crb(frac):
        sig = np.concatenate([np.full(len(ts), frac * (c.max() - c.min())) for c in base])
        sw = S / sig[:, None]
        return np.sqrt(np.diag(np.linalg.inv(sw.T @ sw))) * 100

    assert np.allclose(crb(0.02), 2 * crb(0.01), rtol=1e-9)
    assert np.allclose(crb(0.005), 0.5 * crb(0.01), rtol=1e-9)


def test_more_samples_cannot_loosen_the_bound():
    """MR5: information is monotone in the number of samples, so the bound cannot grow with K."""
    coarse = fisher(NOMINAL_THETA, 0.6, K=20)
    fine = fisher(NOMINAL_THETA, 0.6, K=60)
    assert np.all(fine["crb"] <= coarse["crb"] * (1 + 1e-9)), (coarse["crb"], fine["crb"])


def test_the_coupled_solver_reduces_to_the_reduced_one_at_zero_coupling():
    """MR6: the Pn-carrying solver must be bit-identical to the Pn=0 path it generalises."""
    a = solve_luikov_fd(LuikovParams(*NOMINAL_THETA), n_x=101, n_t=201)
    b = solve_luikov_fd(LuikovParams(*NOMINAL_THETA, Pn=0.0), n_x=101, n_t=201)
    assert np.array_equal(a.T, b.T)
    assert np.array_equal(a.M, b.M)


@pytest.mark.slow
def test_temporal_discretisation_converges_at_first_order():
    """MR7: the scheme is implicit in time, so halving dtau should roughly halve the error.

    Measured against the finest grid as reference. A scheme that stopped converging, or converged
    at the wrong rate, would indicate the time stepping had been broken.
    """
    ref = solve(NOMINAL_THETA, 0.6, 101, 1601)
    errs = []
    for n_t in (201, 401, 801):
        s = solve(NOMINAL_THETA, 0.6, 101, n_t)
        idx = np.linspace(0, len(s.tau) - 1, 21).astype(int)
        ridx = np.linspace(0, len(ref.tau) - 1, 21).astype(int)
        errs.append(float(np.abs(s.M[idx].mean(1) - ref.M[ridx].mean(1)).max()))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(r > 1.5 for r in ratios), f"errors must shrink with refinement: {errs}"
    assert all(1.5 < r < 3.0 for r in ratios), f"expected ~2x per halving, got {ratios}"
