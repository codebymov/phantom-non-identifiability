"""Tangent-safe surrogates: audit tooling for parameter- and input-to-observable tangent maps.

Successor to the phantom-non-identifiability study. The premise: a simulator used for inference,
sensor design or control must preserve not only the state map H(w) but the tangent maps
J_theta = dH/dtheta and J_u = dH/du, and standard training supervises only the first.
"""
__all__ = ["core", "audit"]
