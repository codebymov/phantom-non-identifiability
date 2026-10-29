# Phantom Non-Identifiability

Code and artefacts for **"Phantom Non-Identifiability: When Simplifying a Coupled Simulator
Fabricates an Observability Boundary"** (Sim2Science @ NeurIPS 2026).

A reduction that severs the only path from a parameter to an observed field does not merely lose
accuracy — it manufactures a structural non-identifiability the real system does not have, together
with a confident prescription to buy an instrument the real system may not need. This repository
reproduces every number in the paper.

## What is here

```
paper/      camera-ready PDF and its full LaTeX source (rebuild: cd paper && ./build.sh observability_sim2science)
theory/     identifiability, Fisher/CR, silent band, OSSE, crossover diagnostic
control/    coupled plant, input-Jacobian sweep, closed-loop MPC study
tangentsafe/ surrogate tangent-map audit and the loss/architecture arms
src/kibapap/ the Luikov finite-difference solver everything is built on
runs/       the JSON artefact behind every published number
```

Licensed MIT (see `LICENSE`).

## Install

No GPU is needed; everything in the paper runs on a laptop CPU in minutes.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .          # numpy, scipy, matplotlib; torch only for the PINN arms
```

PyTorch is needed only for the surrogate experiments (App. A, App. C) and is CPU-only:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If small-batch CPU torch oversubscribes threads, cap them — it is several times faster:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
```

All drivers are run from the repository root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. .venv/bin/python theory/posnov_identifiability_sweep.py
```

## Project discipline

**No number appears in the paper that is not written to a JSON artefact by the script that produced
it.** The artefacts are committed under `runs/`, so results can be checked without re-running
anything. The table below is the full map; every entry was verified against the submitted PDF.

| Paper location | Claim | Script | Artefact (key) |
|---|---|---|---|
| Table 1 | CR(Ko), CR(Bi_t) vs Pn; rank 2 at Pn=0 | `theory/posnov_identifiability_sweep.py` | `runs/theory/fisher_nullspace.json` (`rows[].crb`, `rows[].col_norm`) |
| Table 1 | gain row (34×, 14×, 5.1×, 8.1×, 1.2×) | `theory/temperature_channel_value.py` | `runs/theory/temperature_channel_value.json` (`rows[].cr_Ko_M / cr_Ko_MT`) |
| Sec. 3 | E-optimal λ_min 0 → 3.2×10² (mean ⟨T⟩ channel) | `theory/temperature_channel_value.py` | same (`rows[0].lam_min_MT` = 317.3) |
| Sec. 3, App. A | two-head control: 2.498 vs 2.485 (1.2%), 0.022 | `theory/pinn_sensitivity_control.py` | `runs/theory/pinn_sensitivity_control.json` (`sensitivities`) |
| App. A | surrogate rel-L2 0.012 ⟨M⟩, 0.111 ⟨T⟩ | same | same (`fwd_relL2`) |
| Sec. 2 | flux identity 0.4% (Pn≤0.2) → 2.6% (Pn=2) | `theory/flux_identity_check.py` | `runs/theory/flux_identity.json` (`summary`) |
| Sec. 4 | CR slopes −0.959, −0.990 | `theory/weak_coupling_scaling.py` | `runs/theory/weak_coupling.json` (`slopes`) |
| Sec. 4 | probe error 4.5% / 15.5% / 18.7% | same | same (`corners`, split on `pred` < 0.05) |
| Sec. 4 | crossover median 0.037, IQR [0.019, 0.089], 88% below 0.2 | same | same (`sobol`) |
| Sec. 4, Fig. 1 | nominal crossover Pn\* = 0.099 | same (`crossover()` at nominal θ) | derived; sweep in same file |
| Sec. 5, Fig. 1b | silent to Pn=0.35, p=0.44, 0.2% vs 11.9%, Lu 0.5→0.33 | `theory/silent_regime.py` | `runs/theory/silent_regime.json` (`pn_silent`, `rows`) |
| Sec. 5 | Jacobian subspace rotates 68° | `tangentsafe/experiments/m0_reproduce_and_metrics.py` | `runs/tangentsafe/m0b_calibrated.json` (`rows["0.35"].subspace_angle_deg`) |
| Table 2 (App. B) | OSSE recovery, all cells | `theory/posnov_osse_extended.py` | `runs/theory/osse_extended.json` (`results`) |
| Sec. 6 | 9.6% with the added T channel | same | same (`misspec`) |
| App. B | conditional CR 0.4% vs MC 0.37% | same | same (`cr_vs_mc`) |
| App. B | withheld T: 0.0% / 2.1% / 48.9% | same | same (`withheld_T`) |
| App. B | joint 5-parameter: 45%, 41%, 18%, 1.8% | same | same (`joint5`) |
| Table 5 (App. B) | joint 4-parameter recovery, Pn known | `theory/joint4_known_pn.py` | `runs/theory/joint4_known_pn.json` (`cells`) |
| Table 3 (App. C) | five loss/architecture arms, 3 seeds | `tangentsafe/experiments/e1_objective_impossibility.py` | `runs/tangentsafe/e1_impossibility.json` (`arms`) |
| Table 4 (App. E) | closed-loop J, regret, u ratios | `control/structural_mpc.py` | `runs/control/structural_mpc.json` (`results[pn][arm].metrics`) |
| App. E | multistart check: no cost moves > 0.05% | same, single-start run | `runs/control/structural_mpc_singlestart.json` |
| App. E | ‖∂T(0,τ)/∂u‖ = 9.0; slopes +0.994, +0.995 | `control/input_jacobian_sweep.py` | `runs/control/input_jacobian.json` (`zero_check`, `slopes`) |
| Fig. 1 | — | `theory/make_posnov_figure.py` | reads `weak_coupling.json`, `silent_regime.json` |
| Fig. 2 | — | `control/make_control_figure.py` | reads `input_jacobian.json`, `structural_mpc.json` |

Both figures regenerate byte-identically from the committed artefacts, and the PDF rebuilds from
`paper/` alone (19 pages, no overfull boxes, no undefined references).

## Reproducing from scratch

Fast (seconds to ~2 minutes each, no torch):

```bash
PYTHONPATH=. .venv/bin/python theory/posnov_identifiability_sweep.py   # Table 1
PYTHONPATH=. .venv/bin/python theory/temperature_channel_value.py      # Table 1 gain row, E-optimal
PYTHONPATH=. .venv/bin/python theory/weak_coupling_scaling.py          # slopes, probe, Sobol
PYTHONPATH=. .venv/bin/python theory/flux_identity_check.py            # conservation check
PYTHONPATH=. .venv/bin/python theory/silent_regime.py                  # silent band
PYTHONPATH=. .venv/bin/python theory/posnov_osse_extended.py           # Table 2, App. B
PYTHONPATH=. .venv/bin/python control/input_jacobian_sweep.py          # input Jacobian
```

Slower (tens of minutes):

```bash
PYTHONPATH=. .venv/bin/python theory/joint4_known_pn.py                # Table 5  (~30 min)
PYTHONPATH=. .venv/bin/python control/structural_mpc.py                # Table 4  (~30 min)
```

Requires torch (hours; GPU helps):

```bash
PYTHONPATH=. .venv/bin/python tangentsafe/experiments/e1_objective_impossibility.py   # Table 3
PYTHONPATH=. .venv/bin/python theory/pinn_sensitivity_control.py                      # two-head control
```

Self-tests for the solvers:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
PYTHONPATH=. .venv/bin/python control/coupled_plant.py   # bit-identity with the reduced plant at Pn=0
PYTHONPATH=. .venv/bin/python control/luikov_plant.py    # agreement with the benchmark FD solver
```

## Notes and known limits

* **The observation model lives in one place.** `theory/observe.py` defines the channels, the
  per-channel noise and the Fisher/CR machinery, and every analysis script imports it, so the
  observation operator cannot drift between the Fisher analysis and the estimator experiments.
* **`solve()` binds its grid defaults at definition time.** Reassigning `theory.observe.NX` /
  `NT` after import does *not* change them; pass `n_x` / `n_t` explicitly. An earlier
  grid-convergence claim in the paper was withdrawn because of exactly this trap.
* **The gravimetric channel uses a plain nodal mean** of `M`, not a trapezoid quadrature of
  ∫M dξ. This is what the reported numbers refer to; `theory/flux_identity_check.py` reports both
  and shows the trapezoid version is the grid-converged one.
* All results are in silico from one solver family. See the paper's Limitations.
