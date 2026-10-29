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
    """Moisture-only Fisher results at each Posnov number Table 1 reports.

    Session-scoped: a Fisher evaluation costs eight solver calls, so these are computed once
    and shared across every test that needs them.

    Returns:
        dict: Posnov number -> the `theory.observe.fisher` result dict at the nominal theta.
    """
    from tests.paper_values import NOMINAL_THETA
    return {pn: fisher(NOMINAL_THETA, pn) for pn in PAPER_PN}


@pytest.fixture(scope="session")
def fisher_with_t() -> dict:
    """Moisture-plus-<T> Fisher results, for the Table 1 gain row and the E-optimal step.

    Returns:
        dict: Posnov number -> the `theory.observe.fisher` result dict with the optional
            mean-temperature channel included.
    """
    from tests.paper_values import NOMINAL_THETA
    return {pn: fisher(NOMINAL_THETA, pn, with_T=True) for pn in PAPER_PN}


@pytest.fixture(scope="session")
def artefacts(root: Path) -> dict:
    """Every committed JSON artefact, loaded once and keyed by its path relative to ``runs/``.

    Args:
        root: Repository root, from the `root` fixture.

    Returns:
        dict: Path relative to ``runs/`` -> parsed JSON, or ``None`` for a file that failed to
            parse. Invalid JSON is recorded rather than raised so that
            `test_every_artefact_is_valid_json` can report all offenders at once instead of
            dying on the first.
    """
    out = {}
    for p in sorted((root / "runs").rglob("*.json")):
        try:
            out[str(p.relative_to(root / "runs"))] = json.loads(p.read_text())
        except json.JSONDecodeError:
            out[str(p.relative_to(root / "runs"))] = None
    return out


def rounds_to(value: float, printed: float) -> bool:
    """Whether `value` rounds to `printed` at the precision `printed` is written to.

    The paper prints "66%", so 66.26 passes and 66.9 does not. Tolerances in the regression
    tests are the paper's printed precision, not the computation's precision.

    Args:
        value: The computed quantity.
        printed: The quantity as the paper prints it. An ``int`` means zero decimals, so
            ``176`` accepts anything rounding to 176; a ``float`` carries its own precision,
            so ``1.2`` accepts anything rounding to 1.2.

    Returns:
        bool: ``True`` when the two agree at that precision.
    """
    if isinstance(printed, int):
        decimals = 0
    else:
        text = f"{printed!r}"
        decimals = len(text.split(".")[1]) if "." in text else 0
    return round(float(value), decimals) == round(float(printed), decimals)
