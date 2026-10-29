"""Standing tests for the tangent interface (Section 5.4).

Run: PYTHONPATH=. .venv/bin/python tangentsafe/tests/test_core.py
"""
from __future__ import annotations

import numpy as np
import torch

from kibapap.physics import LuikovParams, solve_luikov_fd
from tangentsafe.core.solvers.luikov_torch import solve_luikov_torch
from tangentsafe.core.tangent import NOMINAL, jacobian_ad, jacobian_fd, jacobian_u_ad, jvp

MIRROR_TOL = 1e-11      # mirror vs LAPACK reference: machine precision, NOT bit-identity
ADFD_TOL = 1e-5         # AD vs central differences: limited by FD truncation O(d^2)


def test_mirror_matches_reference():
    """The torch mirror reproduces the untouched scipy reference to machine precision.

    Deliberately NOT bit-identity: the reference uses LAPACK banded solves and the mirror uses LU,
    so operation order differs. Bit-identity at Pn=0 remains asserted against the *original* solver
    by control/coupled_plant.py; this is the mirror's separate, weaker, honest guarantee.
    """
    for pn in (0.0, 0.3, 0.6, 2.0):
        ref = solve_luikov_fd(LuikovParams(0.5, 2.0, 1.0, 1.0, Pn=pn), n_x=101, n_t=201)
        T, M = solve_luikov_torch(0.5, 2.0, 1.0, 1.0, pn)
        eT = np.linalg.norm(T.numpy() - ref.T) / np.linalg.norm(ref.T)
        eM = np.linalg.norm(M.numpy() - ref.M) / np.linalg.norm(ref.M)
        assert eT < MIRROR_TOL and eM < MIRROR_TOL, (pn, eT, eM)
    print("  mirror vs reference: OK (<= 1e-11 rel-L2 across Pn in {0, 0.3, 0.6, 2})")


def test_ad_matches_fd():
    """Reverse-mode AD agrees with the central-difference oracle the predecessor paper used."""
    for pn in (0.0, 0.2, 0.6):
        th = dict(NOMINAL, Pn=pn)
        Ja, Jf = jacobian_ad(th), jacobian_fd(th)
        rel = np.linalg.norm(Ja - Jf) / max(np.linalg.norm(Jf), 1e-300)
        assert rel < ADFD_TOL, (pn, rel)
    print("  AD vs FD Jacobian: OK (<= 1e-5 rel; residual is FD truncation, not AD error)")


def test_forward_and_reverse_agree():
    """Forward-mode JVP and reverse-mode agree to near machine precision."""
    th = dict(NOMINAL, Pn=0.6)
    J_jvp = np.stack([jvp(th, {n: 1.0}) for n in ("Lu", "Ko", "Bi_t", "Bi_m")], axis=1)
    rel = np.linalg.norm(J_jvp - jacobian_ad(th)) / np.linalg.norm(jacobian_ad(th))
    assert rel < 1e-9, rel
    print(f"  forward vs reverse mode: OK ({rel:.2e})")


def test_phantom_null_reproduces_under_ad():
    """The predecessor's central claim, re-derived through autograd rather than differences."""
    J = jacobian_ad(dict(NOMINAL, Pn=0.0))
    cn = np.linalg.norm(J, axis=0)          # [Lu, Ko, Bi_t, Bi_m]
    assert cn[1] == 0.0 and cn[2] == 0.0, cn
    assert cn[0] > 1.0 and cn[3] > 1.0, cn
    print("  phantom null under AD: OK (Ko, Bi_t columns exactly 0.0; Lu, Bi_m nonzero)")


def test_input_tangent_null():
    """The control-side claim: the actuator has exactly zero authority over drying at Pn=0."""
    assert np.linalg.norm(jacobian_u_ad(dict(NOMINAL, Pn=0.0))) == 0.0
    assert np.linalg.norm(jacobian_u_ad(dict(NOMINAL, Pn=0.6))) > 0.1
    print("  input tangent null: OK (||dH/du|| exactly 0 at Pn=0, nonzero at Pn=0.6)")


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in (test_mirror_matches_reference, test_ad_matches_fd, test_forward_and_reverse_agree,
               test_phantom_null_reproduces_under_ad, test_input_tangent_null):
        fn()
    print("\nall standing tests passed")
