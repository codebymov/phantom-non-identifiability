"""Every number the paper prints, in one machine-readable place.

This module is the paper's claims as data. The regression tests in test_published_numbers.py
check each entry two ways:

  * the cheap ones are RECOMPUTED from the solver and compared here, so a change to the
    observation model or the solver that moves a published number fails the suite;
  * the expensive ones are read from their committed JSON artefact and compared here, so
    regenerating an artefact with different numbers also fails the suite.

If a number here is changed, the paper must change with it. That is the point: the paper and the
code cannot drift apart silently, which is exactly what happened to the withdrawn
grid-convergence claim (see README, "Notes and known limits").

Tolerances are the printed precision of the paper, not the precision of the computation: the
paper prints "66%", so the test allows anything that rounds to 66%.
"""
from __future__ import annotations

# theta = (Lu, Ko, Bi_t, Bi_m); the nominal operating point used throughout the paper
NOMINAL_THETA = (0.5, 2.0, 1.0, 1.0)
K_SAMPLES = 40
NOISE_FRAC = 0.01

# ---------------------------------------------------------------- Table 1 (Sec. 4)
# Moisture-only Cramer-Rao relative std [%] at the nominal theta, per Posnov number.
TABLE1_CR_KO = {0.05: 176, 0.1: 99, 0.2: 66, 0.6: 19, 1.0: 1.2}
TABLE1_CR_BIT = {0.05: 259, 0.1: 134, 0.2: 77, 0.6: 26, 1.0: 3.2}
# Gain = moisture-only CR / moisture-plus-<T> CR, on Ko. Non-monotone by construction.
TABLE1_GAIN_KO = {0.05: 34, 0.1: 14, 0.2: 5.1, 0.6: 8.1, 1.0: 1.2}
# rank(S) read from the scaled column norms
RANK_AT_PN0 = 2
RANK_ABOVE_PN0 = 4
# Sec. 3: an E-optimal step lifts lambda_min from 0 to 3.2e2 by adding the mean-<T> channel
E_OPTIMAL_LAMBDA_MIN_PN0 = 3.2e2

# ---------------------------------------------------------------- Sec. 2 (flux identity)
# Discrete q_m(0,tau) vs d<M>/dtau on the paper's grid, with the plain nodal mean <M>.
FLUX_IDENTITY_MAX_PCT_PN_LE_0P2 = 0.4
FLUX_IDENTITY_MAX_PCT_PN_LE_2 = 2.6

# ---------------------------------------------------------------- Sec. 4 (derived scaling)
# Log-log slopes of the two CR bounds over Pn in [1e-3, 1e-1], against -1 from the expansion.
CR_SLOPE_KO = -0.959
CR_SLOPE_BIT = -0.990
# One-probe crossover prediction vs bisection over the 16 box corners, split on predicted Pn*.
PROBE_ERR_MEDIAN_WEAK_PCT = 4.5          # Pn*_pred < 0.05
PROBE_ERR_MAX_WEAK_PCT = 15.5
PROBE_ERR_MEDIAN_STRONG_PCT = 18.7       # Pn*_pred >= 0.05
# Crossover distribution over 256 Sobol points
CROSSOVER_MEDIAN = 0.037
CROSSOVER_IQR = (0.019, 0.089)
CROSSOVER_FRAC_BELOW_0P2 = 0.88
CROSSOVER_AT_NOMINAL = 0.0996
# Collinearity of the {Ko, Bi_t} estimates at Pn = 0.2
CORR_KO_BIT_AT_PN0P2 = 0.997

# ---------------------------------------------------------------- Sec. 5 (the silent band)
SILENT_BAND_TOP = 0.35                   # chi^2 GOF still passes at alpha = 0.05 up to here
SILENT_MEDIAN_P_AT_TOP = 0.44
CALIBRATED_DISCREPANCY_PCT = 0.2         # down from the nominal figure below
NOMINAL_DISCREPANCY_PCT = 11.9
LU_PULLED_TO = 0.33                      # from 0.5
SUBSPACE_ROTATION_DEG = 68
DISCREPANCY_AT_PN0P6_PCT = 17          # reduced model at true parameters, Pn = 0.6

# ---------------------------------------------------------------- Table 2 (App. B, OSSE)
# (value, relative-error-%) per arm, for Ko (true 2.0) and Bi_t (true 1.0)
TABLE2 = {
    "full_matched": {"Ko": (2.00, 0.1), "Bit": (1.00, 0.0)},
    "reduced_matched": {"Ko": (1.00, 50.0), "Bit": (0.50, 50.0)},   # frozen at initialisation
    "full_refined": {"Ko": (2.01, 0.6), "Bit": (1.00, 0.1)},
    "full_gainbias": {"Ko": (1.75, 12.7), "Bit": (0.98, 2.2)},
}
MISSPEC_KO_WITH_T_PCT = 9.6              # the T channel only improves 12.7% to this

# ---------------------------------------------------------------- App. B (validation)
CONDITIONAL_CR_KO_PCT = 0.4
MONTE_CARLO_SPREAD_KO_PCT = 0.37
WITHHELD_T_FULL_PCT = 0.0
WITHHELD_T_MISSPEC_PCT = 2.1
WITHHELD_T_REDUCED_PCT = 48.9
JOINT5_ERR_PCT = {"Lu": 33.1, "Ko": 18, "Bi_t": 45, "Bi_m": 41, "Pn": 1.8}

# ---------------------------------------------------------------- Table 3 (App. C, E1 arms)
# leakage = severed-pathway sensitivity as a fraction of the intact Ko -> <T> pathway [%]
TABLE3_LEAKAGE_PCT = {
    "baseline": 7.3,          # median; one of three seeds does not converge
    "gpinn": 5.8,
    "adaptive": 1.1,
    "causal": 5.7,
    "sharedtrunk": 5.9,
}
TABLE3_BASELINE_PER_SEED_PCT = (1.8, 7.3, 137.5)
SOLVER_TRUE_DM_DLOGKO_AT_PN0P6 = 0.372   # the true coupled value the arms are compared against
BEST_ARM_RESIDUE_FRAC_OF_TRUE_PCT = 7

# ---------------------------------------------------------------- Table 4 (App. E, control)
# Pn -> (reduced J, matched J, uncertain J, regret x, u reduced/matched, control-sens. T term)
TABLE4 = {
    0.10: (1.007, 0.898, 0.911, 1.1, (1.18, 1.23), 0.025),   # dagger: target unreachable
    0.20: (0.060, 0.042, 0.087, 1.4, (1.16, 1.19), 0.022),
    0.35: (0.751, 0.094, 0.093, 8.0, (1.13, 0.87), 0.019),
    0.50: (3.103, 0.261, 0.246, 11.9, (1.11, 0.63), 0.017),
    0.60: (5.267, 0.420, 0.403, 12.5, (1.09, 0.52), 0.016),
    0.80: (10.460, 0.924, 0.884, 11.3, (1.07, 0.35), 0.016),  # dagger
}
TABLE4_EXCLUDED_FROM_REGRET = (0.10, 0.80)   # moisture target outside the reachable set
ABSTRACT_MAX_REGRET = 12.5                   # must equal max regret over non-excluded rows
MULTISTART_MAX_COST_SHIFT_PCT = 0.05
INPUT_JACOBIAN_DT_DU = 9.0                   # ||dT(0,tau)/du||, Pn = 0
INPUT_JACOBIAN_SLOPE_MBAR = 0.994            # weak-coupling log-log slopes
INPUT_JACOBIAN_SLOPE_RATE = 0.995

# ---------------------------------------------------------------- Table 5 (App. B, B1)
# (Pn, noise) -> {param: median relative error %}, plus the fraction recovering all four
TABLE5_MEDIAN_ERR_PCT = {
    (0.2, 0.005): {"Lu": 2.5, "Ko": 24.9, "Bi_t": 28.1, "Bi_m": 4.2},
    (0.2, 0.01): {"Lu": 4.8, "Ko": 35.8, "Bi_t": 47.6, "Bi_m": 5.0},
    (0.2, 0.02): {"Lu": 8.0, "Ko": 39.9, "Bi_t": 50.0, "Bi_m": 8.3},
    (0.6, 0.005): {"Lu": 11.7, "Ko": 9.5, "Bi_t": 13.7, "Bi_m": 12.6},
    (0.6, 0.01): {"Lu": 17.0, "Ko": 13.2, "Bi_t": 19.7, "Bi_m": 18.7},
    (0.6, 0.02): {"Lu": 31.0, "Ko": 16.9, "Bi_t": 45.2, "Bi_m": 39.8},
    (1.0, 0.005): {"Lu": 1.4, "Ko": 0.5, "Bi_t": 1.0, "Bi_m": 0.9},
    (1.0, 0.01): {"Lu": 2.8, "Ko": 1.1, "Bi_t": 2.0, "Bi_m": 1.8},
    (1.0, 0.02): {"Lu": 5.5, "Ko": 2.2, "Bi_t": 3.9, "Bi_m": 3.6},
}
TABLE5_FRAC_RECOVERED = {
    (0.2, 0.005): 0.15, (0.2, 0.01): 0.10, (0.2, 0.02): 0.05,
    (0.6, 0.005): 0.40, (0.6, 0.01): 0.25, (0.6, 0.02): 0.15,
    (1.0, 0.005): 1.00, (1.0, 0.01): 1.00, (1.0, 0.02): 0.90,
}
TABLE5_FRAC_BASIN = 1.0                  # every start reached the same basin in every cell

# ---------------------------------------------------------------- App. A (surrogate)
SURROGATE = {"width": 96, "depth": 6, "n_coll": 8000, "adam": 12000, "lbfgs": 1500}
SURROGATE_RELL2_M = 0.012
SURROGATE_RELL2_T = 0.111
TWO_HEAD_PINN_DT_DLOGKO = 2.498
TWO_HEAD_SOLVER_DT_DLOGKO = 2.485
TWO_HEAD_REL_PCT = 1.2
TWO_HEAD_PINN_DM_DLOGKO = 0.022          # against the solver's exact zero
