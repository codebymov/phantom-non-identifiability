"""Integration layer: the observation model and the Fisher/Cramer-Rao chain.

theory/observe.py is the hub every analysis script imports, and it was untested. These tests
exercise the whole chain -- solver, channels, per-channel noise, scaled sensitivity, Fisher
information, Cramer-Rao bound -- rather than any one function, because that is where the paper's
risk actually lives: a change here moves every published number at once.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.paper_values import (
    CORR_KO_BIT_AT_PN0P2,
    K_SAMPLES,
    NOISE_FRAC,
    NOMINAL_THETA,
    RANK_ABOVE_PN0,
    RANK_AT_PN0,
)
from theory.observe import FRAC, K_DEFAULT, NAMES4, channels, sample_times, sigmas, solve


def test_module_constants_match_the_paper():
    """The paper states K=40 samples and 1%-of-range per-channel noise."""
    assert K_DEFAULT == K_SAMPLES
    assert pytest.approx(NOISE_FRAC) == FRAC


def test_channels_are_finite_and_correctly_shaped():
    sol = solve(NOMINAL_THETA, 0.6)
    ts = sample_times(sol, K_DEFAULT)
    chans = channels(sol, ts)
    assert len(chans) == 2, "moisture-only is <M> and the surface flux"
    for c in chans:
        assert c.shape == (K_DEFAULT,)
        assert np.all(np.isfinite(c))
    assert len(channels(sol, ts, with_T=True)) == 3, "the optional channel is <T>"


def test_noise_is_per_channel_and_scales_with_range():
    """Each sigma is a fixed fraction of its own channel's range, not one shared absolute sigma.

    This is the fix the paper describes: a moisture channel and a moisture-rate channel have
    different units, so one absolute sigma for both is not a well-posed likelihood.
    """
    sol = solve(NOMINAL_THETA, 0.6)
    ts = sample_times(sol, K_DEFAULT)
    chans = channels(sol, ts)
    sig = sigmas(chans)
    assert len(sig) == len(chans)
    for s, c in zip(sig, chans):
        assert s == pytest.approx(FRAC * (c.max() - c.min()))
    assert sig[0] != pytest.approx(sig[1]), "the two channels must not share one absolute sigma"
    # linear in the requested fraction
    assert sigmas(chans, frac=2 * FRAC)[0] == pytest.approx(2 * sig[0])


def test_structural_null_at_pn_zero_is_exact(fisher_moisture):
    """At Pn=0 the Ko and Bi_t sensitivity columns are exactly zero -- the paper's core claim.

    Not small: zero. This is the one assertion that, if it ever fails, invalidates the paper.
    """
    r = fisher_moisture[0.0]
    ko, bit = NAMES4.index("Ko"), NAMES4.index("Bi_t")
    assert r["col_norm"][ko] == 0.0
    assert r["col_norm"][bit] == 0.0
    assert r["rank"] == RANK_AT_PN0
    assert not np.isfinite(r["crb"][ko])
    assert not np.isfinite(r["crb"][bit])
    # the identifiable pair is unaffected
    assert r["col_norm"][NAMES4.index("Lu")] > 1.0
    assert r["col_norm"][NAMES4.index("Bi_m")] > 1.0


@pytest.mark.parametrize("pn", [0.05, 0.1, 0.2, 0.6, 1.0])
def test_rank_is_full_for_every_tested_nonzero_coupling(fisher_moisture, pn):
    r = fisher_moisture[pn]
    assert r["rank"] == RANK_ABOVE_PN0
    assert np.all(np.isfinite(r["crb"]))


def test_crb_is_consistent_with_the_returned_sensitivity(fisher_moisture):
    """Recompute the bound from the S and sigma that fisher() itself returned.

    A guard against the bound and the sensitivity drifting apart, e.g. if the noise model were
    applied in one place and not the other.
    """
    r = fisher_moisture[0.6]
    sw = r["S"] / np.asarray(r["sigma"])[:, None]   # sigma is per observation row
    expected = np.sqrt(np.diag(np.linalg.inv(sw.T @ sw))) * 100
    assert np.allclose(r["crb"], expected, rtol=1e-9)


def test_adding_the_temperature_channel_never_loosens_the_bound(fisher_moisture, fisher_with_t):
    """Fisher information is monotone in the observation set, so the CR bound cannot increase."""
    for pn, m in fisher_moisture.items():
        t = fisher_with_t[pn]
        for j in range(len(NAMES4)):
            if np.isfinite(m["crb"][j]):
                assert t["crb"][j] <= m["crb"][j] * (1 + 1e-9), (pn, NAMES4[j])
            else:
                assert np.isfinite(t["crb"][j]), (pn, NAMES4[j])


def test_sensitivity_grows_and_bound_falls_with_coupling(fisher_moisture):
    """O(Pn) sensitivities and an O(1/Pn) bound, over the weak-coupling range of Eq. 3."""
    ko = NAMES4.index("Ko")
    weak = [0.05, 0.1, 0.2]
    cols = [fisher_moisture[p]["col_norm"][ko] for p in weak]
    crs = [fisher_moisture[p]["crb"][ko] for p in weak]
    assert cols == sorted(cols), "scaled sensitivity must grow with the coupling"
    assert crs == sorted(crs, reverse=True), "the bound must fall as the coupling grows"


def test_estimates_of_ko_and_bit_are_strongly_collinear_at_low_coupling(fisher_moisture):
    """Sec. 4: |corr(Ko, Bi_t)| = 0.997 at Pn = 0.2 -- weak but real, not a null."""
    r = fisher_moisture[0.2]
    cov = np.linalg.inv(r["F"])
    i, j = NAMES4.index("Ko"), NAMES4.index("Bi_t")
    corr = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    assert abs(corr) == pytest.approx(CORR_KO_BIT_AT_PN0P2, abs=5e-4)
