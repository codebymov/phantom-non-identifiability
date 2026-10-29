"""Shared fixtures.

The expensive object in this codebase is a solver call, and a Fisher evaluation costs eight of
them (central differences over four parameters). Fixtures here are session-scoped so a Fisher
result is computed once and reused across every test that needs it, which keeps the default
(non-slow) tier to a few seconds rather than a few minutes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:            # so `theory`, `control`, `tangentsafe` import as packages
    sys.path.insert(0, str(ROOT))

from theory.observe import fisher  # noqa: E402  (needs the sys.path line above)

PAPER_PN = [0.0, 0.05, 0.1, 0.2, 0.6, 1.0]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fisher_moisture() -> dict:
    """Moisture-only Fisher results at each Posnov number Table 1 reports."""
    from tests.paper_values import NOMINAL_THETA
    return {pn: fisher(NOMINAL_THETA, pn) for pn in PAPER_PN}


@pytest.fixture(scope="session")
def fisher_with_t() -> dict:
    """Moisture-plus-<T> Fisher results, for the Table 1 gain row and the E-optimal step."""
    from tests.paper_values import NOMINAL_THETA
    return {pn: fisher(NOMINAL_THETA, pn, with_T=True) for pn in PAPER_PN}


@pytest.fixture(scope="session")
def artefacts(root: Path) -> dict:
    """Every committed JSON artefact, keyed by its path relative to runs/."""
    out = {}
    for p in sorted((root / "runs").rglob("*.json")):
        try:
            out[str(p.relative_to(root / "runs"))] = json.loads(p.read_text())
        except json.JSONDecodeError:
            out[str(p.relative_to(root / "runs"))] = None
    return out


def rounds_to(value: float, printed: float) -> bool:
    """True if `value` rounds to `printed` at the precision `printed` is written to.

    The paper prints "66%", so 66.26 passes and 66.9 does not. Tolerances in the regression
    tests are the paper's printed precision, not the computation's precision.
    """
    if isinstance(printed, int):
        decimals = 0
    else:
        text = f"{printed!r}"
        decimals = len(text.split(".")[1]) if "." in text else 0
    return round(float(value), decimals) == round(float(printed), decimals)
