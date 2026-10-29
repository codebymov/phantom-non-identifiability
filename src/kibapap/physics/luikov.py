"""Dimensionless Luikov system: parameter container and analytical conventions.

The 1D dimensionless system solved across the codebase is

    dT/dtau  =  d2T/dxi2 + Ko * dM/dtau                  in (0, 1) x (0, tau_final)
    dM/dtau  =  Lu * d2M/dxi2  +  Lu * Pn * d2T/dxi2

with Robin BC at the heated surface xi = 0,

    dT/dxi |_(xi=0) = Bi_t * (T - 1)
    dM/dxi |_(xi=0) = Bi_m * M

Neumann BC at the back xi = 1 (insulated / symmetry),

    dT/dxi |_(xi=1) = 0
    dM/dxi |_(xi=1) = 0

and initial conditions T(xi, 0) = 0, M(xi, 0) = 1.

Conventions:
  T : dimensionless temperature, 0 = initial cold web, 1 = hot air bulk
  M : dimensionless moisture,    1 = initial wet web, 0 = equilibrium with air
  xi : dimensionless thickness coordinate, 0 = hot surface, 1 = back
  tau : dimensionless time, scaled by thermal diffusivity / thickness^2

The thermogradient (Soret) cross-term in the moisture equation is controlled by the Posnov
number Pn and defaults to Pn = 0 (the common simplification for fibrous media at moderate
Biot numbers; see Manavi 2024 sec. 2, Barnard 2024 eq. 5). Setting Pn > 0 restores the full
Luikov coupling dM/dtau = Lu d2M/dxi2 + Lu Pn d2T/dxi2. The sign/placement follows the
canonical dimensionless Luikov system (Mikhailov & Ozisik 1984; Cotta 1993; see Lugon &
Silva Neto, COBEM 2003, eqs. 1a-1b), transformed into this file's convention M = 1 - theta_2
(theta_2 the deficit-moisture variable of that reference). When phase 4 introduces web motion
an additional Pe * d/dxi_machine convective term appears.
"""

from dataclasses import dataclass


# frozen: these are physical parameters, i.e. a value object. Freezing makes the shared
# LuikovParams() default in the control plants safe, and makes instances hashable so solver
# calls can be memoised. Nothing in the codebase mutated an instance.
@dataclass(frozen=True)
class LuikovParams:
    """Dimensionless governing parameters for the 1D Luikov system.

    Attributes:
    ----------
    Lu : float
        Luikov number, ratio of mass to thermal diffusivity (alpha_m / alpha_t).
        Typical paper / nonwoven range: 0.1 to 1.0.
    Ko : float
        Kossovich number, ratio of latent to sensible heat capacity for the moisture
        change between initial and equilibrium. Typical range: 0.5 to 5.
    Bi_t : float
        Thermal Biot number at the heated surface.  Typical range: 0.5 to 5.
    Bi_m : float
        Mass Biot number at the heated surface. Typical range: 0.5 to 10.
    Pn : float
        Posnov number (thermogradient / Soret coefficient). Couples moisture to the
        temperature Laplacian: dM/dtau = Lu d2M/dxi2 + Lu Pn d2T/dxi2. Defaults to 0.0,
        which is bit-for-bit the previous decoupled model. Physical drying range 0.2 to 2
        (Cotta 1993 validation case: Pn = 0.6). Kept out of `as_tuple` so the existing
        four-parameter identifiability/sensor-design callers are unaffected.
    """

    Lu: float = 0.5
    Ko: float = 2.0
    Bi_t: float = 1.0
    Bi_m: float = 1.0
    Pn: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.Lu, self.Ko, self.Bi_t, self.Bi_m


# Conventional reference parameter set used as the phase-1 baseline.
# These produce a moderately steep front around tau = 0.4 and equilibrate by tau = 2.
BASELINE = LuikovParams(Lu=0.5, Ko=2.0, Bi_t=1.0, Bi_m=1.0)

# A "sharp front" set used to stress-test gPINN/Sobolev in phase 2.
SHARP_FRONT = LuikovParams(Lu=0.2, Ko=2.5, Bi_t=2.0, Bi_m=2.0)
