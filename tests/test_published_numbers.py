"""Regression layer: the paper's printed numbers, recomputed or read back from their artefacts.

This is the layer that protects the publication. The repository's central promise is that no
number in the paper is unbacked, and until now nothing enforced it -- the withdrawn
grid-convergence claim is what that costs.

Two kinds of test here:

  * RECOMPUTED -- cheap enough to run live from the solver, so breaking the observation model
    breaks the test. These are the strongest.
  * FROM ARTEFACT -- too slow to recompute in a test suite (minutes to hours), so the committed
    JSON is compared against the paper instead. These catch an artefact being regenerated with
    different numbers, which is the "golden file blindness" failure mode: a diff appears, the
    artefact is refreshed without anyone asking why the numbers moved, and a real regression is
    baked in as the new truth.

Tolerances are the paper's printed precision (see conftest.rounds_to).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests import paper_values as pv
from tests.conftest import rounds_to
from theory.observe import NAMES4

pytestmark = pytest.mark.regression

KO, BIT = NAMES4.index("Ko"), NAMES4.index("Bi_t")


# ------------------------------------------------------------------ RECOMPUTED
@pytest.mark.parametrize("pn", sorted(pv.TABLE1_CR_KO))
def test_table1_cramer_rao_bounds(fisher_moisture, pn):
    r = fisher_moisture[pn]
    assert rounds_to(r["crb"][KO], pv.TABLE1_CR_KO[pn]), (pn, r["crb"][KO])
    assert rounds_to(r["crb"][BIT], pv.TABLE1_CR_BIT[pn]), (pn, r["crb"][BIT])


@pytest.mark.parametrize("pn", sorted(pv.TABLE1_GAIN_KO))
def test_table1_gain_row(fisher_moisture, fisher_with_t, pn):
    gain = fisher_moisture[pn]["crb"][KO] / fisher_with_t[pn]["crb"][KO]
    assert rounds_to(gain, pv.TABLE1_GAIN_KO[pn]), (pn, gain)


def test_gain_row_is_non_monotone_as_the_caption_says(fisher_moisture, fisher_with_t):
    """The caption explains a non-monotonicity; if it ever became monotone the caption is wrong."""
    g = [fisher_moisture[p]["crb"][KO] / fisher_with_t[p]["crb"][KO] for p in (0.2, 0.6)]
    assert g[1] > g[0], "the gain is expected to rise again across the collinearity collapse"


def test_e_optimal_lambda_min_at_pn_zero(fisher_moisture, fisher_with_t):
    """Sec. 3: lambda_min(F) lifts from 0 to 3.2e2 by adding the mean-<T> channel."""
    assert fisher_moisture[0.0]["eig"].min() == pytest.approx(0.0, abs=1e-9)
    lam = fisher_with_t[0.0]["eig"].min()
    assert lam == pytest.approx(pv.E_OPTIMAL_LAMBDA_MIN_PN0, rel=0.02)


def test_flux_identity_holds_to_the_stated_tolerance():
    """Sec. 2's conservation check, at the two bounds the paper quotes."""
    from theory.flux_identity_check import THETA0, residual
    from theory.observe import NT, NX
    lo = max(residual(THETA0, pn, NX, NT) for pn in (0.0, 0.05, 0.1, 0.2))
    hi = residual(THETA0, 2.0, NX, NT)
    assert lo <= pv.FLUX_IDENTITY_MAX_PCT_PN_LE_0P2 + 0.05, lo
    assert rounds_to(hi, pv.FLUX_IDENTITY_MAX_PCT_PN_LE_2), hi


@pytest.mark.slow
def test_crossover_at_the_nominal_point():
    """Sec. 5 and Fig. 1: Pn* = 0.099 at the nominal theta."""
    from theory.weak_coupling_scaling import crossover
    assert rounds_to(crossover(pv.NOMINAL_THETA), pv.CROSSOVER_AT_NOMINAL)


# ------------------------------------------------------------------ FROM ARTEFACT
def test_cr_slopes(artefacts):
    s = artefacts["theory/weak_coupling.json"]["slopes"]
    assert rounds_to(s["cr_Ko"], pv.CR_SLOPE_KO)
    assert rounds_to(s["cr_Bit"], pv.CR_SLOPE_BIT)


def test_one_probe_prediction_error(artefacts):
    """Sec. 4's 4.5% / 15.5% / 18.7%, recomputed from the stored corners.

    The split is on PREDICTED Pn*, which is what a practitioner holding one probe solve has.
    """
    corners = artefacts["theory/weak_coupling.json"]["corners"]
    pred = np.array([c["pred"] for c in corners])
    meas = np.array([c["meas"] for c in corners])
    ok = np.isfinite(meas)
    err = np.abs(pred[ok] / meas[ok] - 1) * 100
    weak = pred[ok] < 0.05
    assert rounds_to(np.median(err[weak]), pv.PROBE_ERR_MEDIAN_WEAK_PCT)
    assert rounds_to(err[weak].max(), pv.PROBE_ERR_MAX_WEAK_PCT)
    assert rounds_to(np.median(err[~weak]), pv.PROBE_ERR_MEDIAN_STRONG_PCT)


def test_crossover_distribution(artefacts):
    s = artefacts["theory/weak_coupling.json"]["sobol"]
    assert rounds_to(s["quantiles"]["p50"], pv.CROSSOVER_MEDIAN)
    assert rounds_to(s["quantiles"]["p25"], pv.CROSSOVER_IQR[0])
    assert rounds_to(s["quantiles"]["p75"], pv.CROSSOVER_IQR[1])
    assert rounds_to(s["frac_below_0p2"], pv.CROSSOVER_FRAC_BELOW_0P2)


def test_silent_band(artefacts):
    d = artefacts["theory/silent_regime.json"]
    assert rounds_to(d["pn_silent"], pv.SILENT_BAND_TOP)
    row = min(d["rows"], key=lambda r: abs(r["pn"] - d["pn_silent"]))
    assert rounds_to(row["p"], pv.SILENT_MEDIAN_P_AT_TOP)
    assert rounds_to(row["relL2_calibrated"] * 100, pv.CALIBRATED_DISCREPANCY_PCT)
    assert rounds_to(row["relL2_nominal"] * 100, pv.NOMINAL_DISCREPANCY_PCT)
    assert rounds_to(row["Lu_fit"], pv.LU_PULLED_TO)
    # the window runs from the crossover to the top of the band, and overlaps the assumed range
    assert pv.CROSSOVER_AT_NOMINAL < pv.SILENT_BAND_TOP
    assert pv.SILENT_BAND_TOP > 0.2


def test_subspace_rotation(artefacts):
    rows = artefacts["tangentsafe/m0b_calibrated.json"]["rows"]
    assert rounds_to(rows["0.35"]["subspace_angle_deg"], pv.SUBSPACE_ROTATION_DEG)
    assert rows["0.35"]["order_agrees"] is False, "the sensitivity ordering is said to invert"


def test_table2_osse(artefacts):
    r = artefacts["theory/osse_extended.json"]["results"]
    cases = {
        "full_matched": r["matched grid|full  (Pn=0.6)"],
        "reduced_matched": r["matched grid|reduced (Pn=0)"],
        "full_refined": r["refined-grid twin|full  (Pn=0.6)"],
        "full_gainbias": r["misspec"]["moisture-only"],
    }
    for name, got in cases.items():
        want = pv.TABLE2[name]
        assert rounds_to(got["Ko"], want["Ko"][0]), (name, got["Ko"])
        assert rounds_to(got["Bit"], want["Bit"][0]), (name, got["Bit"])
    assert cases["reduced_matched"]["Ko_std"] == 0.0, "the reduced model is frozen: zero variance"
    assert rounds_to(r["misspec"]["moisture + mean T"]["Ko_err"], pv.MISSPEC_KO_WITH_T_PCT)


def test_appendix_b_validation(artefacts):
    r = artefacts["theory/osse_extended.json"]["results"]
    assert rounds_to(r["cr_vs_mc"]["conditional"][0], pv.CONDITIONAL_CR_KO_PCT)
    assert rounds_to(r["cr_vs_mc"]["mc"][0], pv.MONTE_CARLO_SPREAD_KO_PCT)
    w = r["withheld_T"]
    assert rounds_to(w["full  (Pn=0.6)"], pv.WITHHELD_T_FULL_PCT)
    assert rounds_to(w["misspecified fit"], pv.WITHHELD_T_MISSPEC_PCT)
    assert rounds_to(w["reduced (Pn=0)"], pv.WITHHELD_T_REDUCED_PCT)
    for k, want in pv.JOINT5_ERR_PCT.items():
        assert rounds_to(r["joint5"][k]["err_pct"], want), k


def test_table3_e1_arms(artefacts):
    arms = artefacts["tangentsafe/e1_impossibility.json"]["arms"]
    ref = artefacts["tangentsafe/e1_impossibility.json"]["solver_reference"]["0.6"]
    assert rounds_to(ref, pv.SOLVER_TRUE_DM_DLOGKO_AT_PN0P6)
    # the baseline is reported as a median because one of its three seeds does not converge
    per = sorted(s["ratio"] * 100 for s in arms["baseline"]["seeds"])
    for got, want in zip(per, sorted(pv.TABLE3_BASELINE_PER_SEED_PCT)):
        assert rounds_to(got, want), (got, want)
    assert rounds_to(np.median(per), pv.TABLE3_LEAKAGE_PCT["baseline"])
    for arm in ("gpinn", "adaptive", "causal", "sharedtrunk"):
        assert rounds_to(arms[arm]["ratio_mean"] * 100, pv.TABLE3_LEAKAGE_PCT[arm]), arm
    # the claim is that none of them approaches the true coupled value
    best = min(arms[a]["dM_mean"] for a in arms)
    assert rounds_to(best / ref * 100, pv.BEST_ARM_RESIDUE_FRAC_OF_TRUE_PCT)


def test_table4_closed_loop(artefacts):
    res = artefacts["control/structural_mpc.json"]["results"]
    for pn, (red, mat, unc, regret, uratio, tterm) in pv.TABLE4.items():
        m = {a: res[f"{pn:g}"][a]["metrics"] for a in ("reduced", "matched", "uncertain")}
        assert rounds_to(m["reduced"]["J_true"], red), pn
        assert rounds_to(m["matched"]["J_true"], mat), pn
        assert rounds_to(m["uncertain"]["J_true"], unc), pn
        assert rounds_to(m["reduced"]["J_true"] / m["matched"]["J_true"], regret), pn
        assert rounds_to(m["reduced"]["u_mean"], uratio[0]), pn
        assert rounds_to(m["matched"]["u_mean"], uratio[1]), pn
        assert rounds_to(m["reduced"]["temp_track"], tterm), pn


def test_abstract_regret_matches_table4(artefacts):
    """The abstract says "up to 12.5x". It must equal the largest non-excluded regret in Table 4.

    The submitted version said 12x, which matched no row. This test is why that cannot recur.
    """
    res = artefacts["control/structural_mpc.json"]["results"]
    regrets = {
        pn: res[f"{pn:g}"]["reduced"]["metrics"]["J_true"] / res[f"{pn:g}"]["matched"]["metrics"]["J_true"]
        for pn in pv.TABLE4
        if pn not in pv.TABLE4_EXCLUDED_FROM_REGRET
    }
    assert rounds_to(max(regrets.values()), pv.ABSTRACT_MAX_REGRET)


def test_control_sensitive_term_improves_while_cost_worsens(artefacts):
    """Sec. 7's silence claim: the term the reduced model can see improves monotonically."""
    res = artefacts["control/structural_mpc.json"]["results"]
    # over the rows the regret claim uses; the two dagger rows have the moisture target outside
    # the reachable set, which breaks the cost trend by construction.
    pns = [p for p in sorted(pv.TABLE4) if p not in pv.TABLE4_EXCLUDED_FROM_REGRET]
    terms = [res[f"{p:g}"]["reduced"]["metrics"]["temp_track"] for p in pns]
    costs = [res[f"{p:g}"]["reduced"]["metrics"]["J_true"] for p in pns]
    assert terms == sorted(terms, reverse=True), "must improve monotonically"
    assert costs == sorted(costs), "while the true cost rises"


def test_multistart_robustness(artefacts):
    """App. E: against a single warm start, no closed-loop cost moves by more than 0.05%."""
    multi = artefacts["control/structural_mpc.json"]["results"]
    single = artefacts["control/structural_mpc_singlestart.json"]["results"]
    worst = max(
        abs(single[pn][a]["metrics"]["J_true"] - multi[pn][a]["metrics"]["J_true"])
        / multi[pn][a]["metrics"]["J_true"] * 100
        for pn in multi for a in multi[pn]
    )
    assert worst < pv.MULTISTART_MAX_COST_SHIFT_PCT, worst


def test_input_jacobian(artefacts):
    d = artefacts["control/input_jacobian.json"]
    assert d["zero_check"]["dM"] == 0.0, "input-to-moisture Jacobian is exactly zero at Pn=0"
    assert d["zero_check"]["dR"] == 0.0
    assert rounds_to(d["zero_check"]["dT"], pv.INPUT_JACOBIAN_DT_DU)
    assert rounds_to(d["slopes"]["dM_weak"], pv.INPUT_JACOBIAN_SLOPE_MBAR)
    assert rounds_to(d["slopes"]["dR_weak"], pv.INPUT_JACOBIAN_SLOPE_RATE)


def test_table5_joint_four_parameter(artefacts):
    cells = artefacts["theory/joint4_known_pn.json"]["cells"]
    by_key = {(c["pn"], c["noise_frac"]): c for c in cells}
    assert set(by_key) == set(pv.TABLE5_MEDIAN_ERR_PCT)
    for key, want in pv.TABLE5_MEDIAN_ERR_PCT.items():
        got = by_key[key]
        for name, w in want.items():
            assert rounds_to(got["median_err_pct"][name], w), (key, name)
        assert got["frac_recovered"] == pytest.approx(pv.TABLE5_FRAC_RECOVERED[key], abs=1e-9)
        assert got["frac_basin"] == pytest.approx(pv.TABLE5_FRAC_BASIN), key


def test_surrogate_and_two_head_control(artefacts):
    cfg = artefacts["parametric/config.json"]
    assert (cfg["hidden"], cfg["depth"]) == (pv.SURROGATE["width"], pv.SURROGATE["depth"])
    assert cfg["n_coll"] == pv.SURROGATE["n_coll"]
    assert (cfg["adam_iters"], cfg["lbfgs_iters"]) == (pv.SURROGATE["adam"], pv.SURROGATE["lbfgs"])
    s = artefacts["theory/pinn_sensitivity_control.json"]
    assert rounds_to(s["fwd_relL2"]["M"], pv.SURROGATE_RELL2_M)
    assert rounds_to(s["fwd_relL2"]["T"], pv.SURROGATE_RELL2_T)
    ko = s["sensitivities"]["Ko|<T>"]
    assert rounds_to(ko["pinn"], pv.TWO_HEAD_PINN_DT_DLOGKO)
    assert rounds_to(ko["fd"], pv.TWO_HEAD_SOLVER_DT_DLOGKO)
    assert rounds_to(ko["rel"] * 100, pv.TWO_HEAD_REL_PCT)
    km = s["sensitivities"]["Ko|<M>"]
    assert rounds_to(km["pinn"], pv.TWO_HEAD_PINN_DM_DLOGKO)
    assert km["fd"] == 0.0, "the solver's moisture-side sensitivity is exactly zero at Pn=0"
