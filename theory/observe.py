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
    """Run the reference finite-difference solver at a parameter point and coupling strength.

    Args:
        theta (sequence of float): The four parameters (Lu, Ko, Bi_t, Bi_m), in that order.
        pn (float): Posnov number. ``0`` is the reduced model, in which the moisture balance is
            autonomous and the phantom null appears.
        n_x (int): Number of spatial nodes, boundaries included. Defaults to ``NX`` (101).
        n_t (int): Number of time levels, ``tau = 0`` included. Defaults to ``NT`` (201).

    Returns:
        FDSolution: Fields ``T`` and ``M`` of shape ``(n_t, n_x)``, with ``xi``, ``tau`` and the
            ``LuikovParams`` used.

    Note:
        ``n_x`` and ``n_t`` are bound as defaults at definition time, so reassigning the module
        constants ``NX`` or ``NT`` after import does **not** change them. Pass them explicitly.
        An earlier grid-convergence claim in the paper was withdrawn because of this.
    """
    return solve_luikov_fd(LuikovParams(*theta, Pn=pn), n_x=n_x, n_t=n_t)


def sample_times(sol, K=K_DEFAULT):
    """Pick the K instants at which the instruments are read.

    Indices are spread evenly over the solver's time grid, excluding ``tau = 0`` where the
    initial condition makes every channel identical by construction.

    Args:
        sol (FDSolution): A solved trajectory, used only for its time grid.
        K (int): Number of sample times. Defaults to ``K_DEFAULT`` (40), the paper's schedule.

    Returns:
        numpy.ndarray: Sample times, shape ``(K,)``, drawn from ``sol.tau``.
    """
    idx = np.linspace(1, len(sol.tau) - 1, K).astype(int)
    return sol.tau[idx]


def channels(sol, t_samp, with_T=False):
    """Evaluate the instrument channels on a trajectory.

    The surface mass flux is taken from the closed-form Robin derivatives rather than by
    numerically differentiating the field, so no finite differencing enters the observation:
    ``q_m(0, tau) = -Lu (Bi_m M(0, tau) + Pn Bi_t (T(0, tau) - 1))``.

    Args:
        sol (FDSolution): A solved trajectory.
        t_samp (numpy.ndarray): Instants to interpolate onto, as returned by `sample_times`.
        with_T (bool): Append the optional mean-temperature channel. Defaults to ``False``,
            which is the moisture-only instrument the paper's central result concerns.

    Returns:
        list of numpy.ndarray: ``[Mbar, q_m]``, each of shape ``(len(t_samp),)``, plus
            ``[Tbar]`` when `with_T` is set. ``Mbar`` is a plain nodal mean of ``M``, not a
            trapezoid quadrature of the integral; that is the operator the paper's numbers refer
            to.
    """
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
    """Noise standard deviation for each channel, as a fraction of that channel's own range.

    One absolute sigma shared across channels would not be a well-posed likelihood here, since a
    moisture channel and a moisture-rate channel carry different units.

    Args:
        chans (list of numpy.ndarray): Channels, as returned by `channels`.
        frac (float): Fraction of each channel's range. Defaults to ``FRAC`` (0.01).

    Returns:
        list of float: One standard deviation per channel, in the same order.

    Note:
        ``frac`` is bound as a default at definition time, so reassigning ``FRAC`` after import
        does not change it. Pass it explicitly.
    """
    return [float(frac * (c.max() - c.min())) for c in chans]


def observe(theta, pn, t_samp=None, with_T=False, K=K_DEFAULT, n_x=NX, n_t=NT):
    """Solve, sample and stack every channel into one observation vector.

    Args:
        theta (sequence of float): The four parameters (Lu, Ko, Bi_t, Bi_m).
        pn (float): Posnov number.
        t_samp (numpy.ndarray, optional): Instants to observe at. Defaults to ``None``, which
            draws them from the solve with `sample_times`.
        with_T (bool): Include the optional mean-temperature channel. Defaults to ``False``.
        K (int): Number of sample times, used only when `t_samp` is ``None``.
        n_x (int): Spatial nodes passed to the solver.
        n_t (int): Time levels passed to the solver.

    Returns:
        tuple: ``(h, ts)`` where ``h`` is the concatenated observation vector of shape
            ``(n_channels * K,)`` and ``ts`` are the sample times actually used.
    """
    sol = solve(theta, pn, n_x, n_t)
    ts = sample_times(sol, K) if t_samp is None else t_samp
    return np.concatenate(channels(sol, ts, with_T)), ts


def sensitivity(theta, pn, with_T=False, K=K_DEFAULT, params=NAMES4, d=D_LOG):
    """Scaled sensitivity ``S_ij = dH_i / dlog theta_j``, by central differences through the solver.

    Sample times are held fixed at the nominal solve, so perturbed runs are compared on the same
    instants rather than on each run's own grid.

    Args:
        theta (sequence of float): The four parameters (Lu, Ko, Bi_t, Bi_m).
        pn (float): Posnov number.
        with_T (bool): Include the optional mean-temperature channel. Defaults to ``False``.
        K (int): Number of sample times. Defaults to ``K_DEFAULT``.
        params (list of str): Which of ``(Lu, Ko, Bi_t, Bi_m, Pn)`` are treated as unknown, and
            therefore which columns ``S`` has. Defaults to the four non-coupling parameters.
        d (float): Relative central-difference step in log-parameter space. Defaults to
            ``D_LOG`` (1e-3).

    Returns:
        tuple: ``(S, sig)`` where ``S`` has shape ``(n_channels * K, len(params))`` and ``sig``
            is the per-observation-row noise standard deviation, shape ``(n_channels * K,)``
            -- already expanded per row, not one entry per channel.
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
    """Fisher information ``F = S^T Sigma^-1 S``, with the structural rank and the marginal bound.

    Rank is read from the noise-scaled column norms rather than from an eigenvalue ratio: a
    column counts as present when ``||S_:j/sigma|| > 1e-9 max_k ||S_:k/sigma||``. An eigenvalue
    cut-off would report weak-but-real coupling as a null, which is the binary verdict this work
    argues against -- practical weakness belongs in the bound, not in the rank.

    Args:
        theta (sequence of float): The four parameters (Lu, Ko, Bi_t, Bi_m).
        pn (float): Posnov number.
        with_T (bool): Include the optional mean-temperature channel. Defaults to ``False``.
        K (int): Number of sample times. Defaults to ``K_DEFAULT``.
        params (list of str): Parameters treated as unknown. Defaults to the four
            non-coupling parameters.

    Returns:
        dict: With keys ``S`` (scaled sensitivity), ``sigma`` (per-row noise std), ``F`` (Fisher
            matrix), ``eig`` (eigenvalues, descending), ``rank`` (structural rank), ``crb``
            (marginal Cramer-Rao relative standard deviation in percent, ``inf`` for every entry
            when the rank is deficient), ``col_norm`` (scaled column norms) and ``params``.
    """
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
    """Marginal Cramer-Rao relative standard deviation for one parameter, in percent.

    Args:
        theta (sequence of float): The four parameters (Lu, Ko, Bi_t, Bi_m).
        pn (float): Posnov number.
        which (str): Parameter to report. Must appear in `params`. Defaults to ``"Ko"``.
        with_T (bool): Include the optional mean-temperature channel. Defaults to ``False``.
        K (int): Number of sample times. Defaults to ``K_DEFAULT``.
        params (list of str): Parameters treated as unknown, which sets what the bound is
            marginal over. Defaults to the four non-coupling parameters.

    Returns:
        float: The bound in percent, or ``inf`` when the parameter is structurally
            unidentifiable at this coupling.

    Raises:
        ValueError: If `which` is not in `params`.
    """
    r = fisher(theta, pn, with_T, K, params)
    if which not in r["params"]:
        raise ValueError(f"{which!r} is not among the estimated parameters {r['params']}")
    return float(r["crb"][r["params"].index(which)])
