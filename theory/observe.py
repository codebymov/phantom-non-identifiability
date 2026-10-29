"""Shared observation model for the drying cell: channels, noise, sensitivity, Fisher.

Every identifiability, OSSE and diagnostic script imports from here so that the observation
operator and the noise model cannot drift apart between the Fisher analysis and the estimator
experiments (they previously did: a flat sigma=0.01 in the Fisher analysis against per-channel
1%-of-range noise in the OSSE, which is not the same likelihood -- and a flat absolute sigma on
both a moisture and a moisture-*rate* channel is dimensionally incoherent).

Channels (what a drying cell actually instruments):

  Mbar(tau) = <M(.,tau)>_xi                      gravimetric scale (through-thickness mean)
  q_m(tau)  = -Lu ( dM/dxi + Pn dT/dxi )|_xi=0   surface mass flux, i.e. the mass-flow meter
  Tbar(tau) = <T(.,tau)>_xi                      optional added temperature channel

The flux is the physically correct mass-flow-meter operator. Writing the moisture balance in
conservation form, d_tau M = -d_xi q_m with q_m = -Lu (d_xi M + Pn d_xi T), so with the Neumann
back boundary q_m(1) = 0 and

    q_m(0, tau) = d<M>/dtau            exactly.

The two moisture channels are therefore a quantity and its exact time derivative; they still carry
independent information because they are separate instruments with independent noise (differencing
a noisy scale trace is not the same measurement as a mass-flow reading). Both boundary derivatives
are available in closed form from the Robin conditions, so no numerical differentiation is needed:

    dM/dxi|_0 = Bi_m M(0,tau),      dT/dxi|_0 = Bi_t (T(0,tau) - 1)
    => q_m(0,tau) = -Lu ( Bi_m M(0,tau) + Pn Bi_t (T(0,tau) - 1) ).

Note the structure this exposes: at Pn = 0 the flux is -Lu Bi_m M(0,tau), autonomous in the
moisture field, so the phantom null on {Ko, Bi_t} survives exactly. For Pn > 0 the flux acquires an
*explicit* Bi_t dependence on top of the implicit one through M.

Noise model: independent Gaussian per channel with std = FRAC of that channel's range, so the
Fisher information is F = S^T Sigma^{-1} S with Sigma = diag(sigma_c^2) blockwise.
"""
from __future__ import annotations

import numpy as np

from kibapap.physics import LuikovParams, solve_luikov_fd

K_DEFAULT = 40
FRAC = 0.01        # sensor noise as a fraction of each channel's own range
NX, NT = 101, 201
D_LOG = 1e-3       # central-difference step in log-parameter space
NAMES4 = ["Lu", "Ko", "Bi_t", "Bi_m"]
NAMES5 = ["Lu", "Ko", "Bi_t", "Bi_m", "Pn"]


def solve(theta, pn, n_x=NX, n_t=NT):
    return solve_luikov_fd(LuikovParams(*theta, Pn=pn), n_x=n_x, n_t=n_t)


def sample_times(sol, K=K_DEFAULT):
    idx = np.linspace(1, len(sol.tau) - 1, K).astype(int)
    return sol.tau[idx]


def channels(sol, t_samp, with_T=False):
    """[Mbar, q_m] (+ [Tbar]) interpolated onto t_samp. q_m from the closed-form Robin fluxes."""
    p = sol.params
    Mbar = np.interp(t_samp, sol.tau, sol.M.mean(1))
    dMdxi0 = p.Bi_m * sol.M[:, 0]
    dTdxi0 = p.Bi_t * (sol.T[:, 0] - 1.0)
    q = -p.Lu * (dMdxi0 + p.Pn * dTdxi0)
    qm = np.interp(t_samp, sol.tau, q)
    out = [Mbar, qm]
    if with_T:
        out.append(np.interp(t_samp, sol.tau, sol.T.mean(1)))
    return out


def sigmas(chans, frac=FRAC):
    """Per-channel noise std = frac * (range of that channel)."""
    return [float(frac * (c.max() - c.min())) for c in chans]


def observe(theta, pn, t_samp=None, with_T=False, K=K_DEFAULT, n_x=NX, n_t=NT):
    """Stacked observation vector, plus the sample times used."""
    sol = solve(theta, pn, n_x, n_t)
    ts = sample_times(sol, K) if t_samp is None else t_samp
    return np.concatenate(channels(sol, ts, with_T)), ts


def sensitivity(theta, pn, with_T=False, K=K_DEFAULT, params=NAMES4, d=D_LOG):
    """Scaled Jacobian S_ij = d H_i / d log theta_j, central differences through the solver.

    `params` selects which of (Lu, Ko, Bi_t, Bi_m, Pn) are treated as unknown. Sample times are
    held fixed at the nominal solve so perturbed runs are compared on the same instants.
    """
    th = list(theta)
    base = solve(th, pn)
    ts = sample_times(base, K)
    cols = []
    for name in params:
        tp, tm, pp, pm = list(th), list(th), pn, pn
        if name == "Pn":
            pp, pm = pn * (1 + d), pn * (1 - d)
        else:
            j = NAMES4.index(name)
            tp[j] *= (1 + d); tm[j] *= (1 - d)
        hp = np.concatenate(channels(solve(tp, pp), ts, with_T))
        hm = np.concatenate(channels(solve(tm, pm), ts, with_T))
        cols.append((hp - hm) / (2 * d))
    S = np.array(cols).T
    sig = np.concatenate([np.full(K, s) for s in sigmas(channels(base, ts, with_T))])
    return S, sig


def fisher(theta, pn, with_T=False, K=K_DEFAULT, params=NAMES4):
    """F = S^T Sigma^-1 S with the per-channel noise model, plus rank and marginal CR (%)."""
    S, sig = sensitivity(theta, pn, with_T, K, params)
    Sw = S / sig[:, None]
    F = Sw.T @ Sw
    w = np.linalg.eigvalsh(F)[::-1]
    cn = np.linalg.norm(Sw, axis=0)

    # STRUCTURAL rank from the scaled columns, not from an eigenvalue ratio. The phantom null is a
    # column that is zero to machine precision (exactly 0.0 at Pn=0); weak coupling gives a small
    # but genuinely nonzero column, whose eigenvalue ratio can fall below any fixed relative
    # tolerance while the parameter is still connected to the observations. Using an eigenvalue
    # cut-off here would re-introduce, as a numerical artefact, exactly the binary rank verdict this
    # paper argues against -- practical weakness must be reported by the CR bound, not by the rank.
    rank = int((cn > 1e-9 * cn.max()).sum()) if cn.max() > 0 else 0
    crb = np.full(len(params), np.inf)
    if rank == len(params):
        try:
            Fi = np.linalg.solve(F, np.eye(len(params)))
            d = np.diag(Fi)
            if np.all(np.isfinite(d)) and np.all(d > 0):
                crb = np.sqrt(d) * 100
        except np.linalg.LinAlgError:
            pass
    return dict(S=S, sigma=sig, F=F, eig=w, rank=rank, crb=crb, col_norm=cn, params=list(params))


def cr(theta, pn, which="Ko", with_T=False, K=K_DEFAULT, params=NAMES4):
    r = fisher(theta, pn, with_T, K, params)
    return float(r["crb"][r["params"].index(which)])
