"""The reproducibility contract, as tests.

The repository promises three things about its artefacts. Each is checked here rather than
trusted:

  1. every artefact the README's traceability table names exists and is valid JSON;
  2. no artefact leaks an absolute path from the machine that produced it;
  3. two scripts never write the same artefact path, which is how Table 1's artefact used to be
     clobberable by running theory/identifiability_fim.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Artefact -> the keys the paper actually reads out of it. Keeping this list here means a
# refactor that renames a key fails a test instead of silently orphaning a published number.
REQUIRED = {
    "theory/fisher_nullspace.json": ["theta0", "rows"],
    "theory/temperature_channel_value.json": ["theta0", "rows"],
    "theory/weak_coupling.json": ["slopes", "corners", "pred_err_pct", "sobol"],
    "theory/silent_regime.json": ["pn_silent", "rows", "alpha", "dof"],
    "theory/osse_extended.json": ["results", "pn_true", "seeds"],
    "theory/joint4_known_pn.json": ["cells", "seeds", "n_starts"],
    "theory/flux_identity.json": ["rows", "summary"],
    "theory/pinn_sensitivity_control.json": ["sensitivities", "fwd_relL2"],
    "theory/sampling_sufficiency.json": ["rows"],
    "tangentsafe/e1_impossibility.json": ["arms", "solver_reference", "setup"],
    "tangentsafe/m0b_calibrated.json": ["rows", "theta"],
    "control/structural_mpc.json": ["results", "setup", "n_starts"],
    "control/structural_mpc_singlestart.json": ["results"],
    "control/input_jacobian.json": ["zero_check", "slopes", "rows"],
    "parametric/config.json": ["hidden", "depth", "n_coll", "adam_iters", "lbfgs_iters"],
}

HOME_PATH = re.compile(r"/home/[A-Za-z0-9_.-]+")


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_artefact_exists_and_has_the_keys_the_paper_reads(root: Path, name: str):
    p = root / "runs" / name
    assert p.is_file(), f"{name} is referenced by the README but missing"
    d = json.loads(p.read_text())
    for key in REQUIRED[name]:
        assert key in d, f"{name} is missing key {key!r}"


def test_no_artefact_contains_an_absolute_home_path(root: Path):
    """The release builder scrubs these; this catches a newly written one before release.

    runs/theory/pinn_sensitivity_control.json records the checkpoint it read, which is how an
    absolute path got in there in the first place.
    """
    offenders = []
    for p in sorted((root / "runs").rglob("*.json")):
        m = HOME_PATH.search(p.read_text())
        if m:
            offenders.append(f"{p.relative_to(root)}: {m.group(0)}")
    assert not offenders, "absolute paths found:\n  " + "\n  ".join(offenders)


def test_no_two_scripts_write_the_same_artefact(root: Path):
    """Two writers for one path means running the wrong script silently replaces a published
    artefact, possibly with a different schema. theory/identifiability_fim.py and
    theory/posnov_identifiability_sweep.py both targeted fisher_nullspace.json until this was
    separated.
    """
    writers: dict[str, list[str]] = {}
    pat = re.compile(r'["\'](runs/[A-Za-z0-9_/]+\.json)["\']')
    for py in sorted(root.rglob("*.py")):
        if any(part in {".diaglib", "dist", "external", ".venv"} for part in py.parts):
            continue
        text = py.read_text()
        if "json.dump" not in text:      # json.dump / json.dumps: a writer serialises, a reader
            continue                     # only names the path it loads
        for target in set(pat.findall(text)):
            writers.setdefault(target, []).append(str(py.relative_to(root)))
    clashes = {k: v for k, v in writers.items() if len(v) > 1}
    assert not clashes, f"artefacts with more than one writer: {clashes}"


def test_every_artefact_is_valid_json(artefacts: dict):
    broken = [k for k, v in artefacts.items() if v is None]
    assert not broken, f"invalid JSON: {broken}"
