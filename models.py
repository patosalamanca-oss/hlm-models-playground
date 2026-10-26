"""Hillslope Link Model (HLM) right-hand-side definitions.

Every model is registered in :data:`MODELS` by name, so the notebook can pick
and test any of them interchangeably::

    from models import MODELS, get_model
    model = get_model("TopLayer")
    sol = solve_ivp(model.dydt, ..., args=(params, global_params, forcing))

Adding a new model
------------------
Each model is built from a *flux function* and a *RHS function*:

* ``fluxes(t, y, params, global_params, forcing) -> dict``
      returns every internal flux/rate (q_pl, q_pt, q_ts, q_sl, evaporation,
      k_t, ...). This is the single source of truth.
* ``dydt(t, y, params, global_params, forcing, parents_q=0, parents_qb=0)``
      assembles dy/dt from those fluxes; this is what ``solve_ivp`` integrates.

Register the pair with :func:`register_model`. Because the RHS and the flux
extractor share the same flux function, the post-run flux analysis is always
consistent with the integrated solution. Use :func:`flux_timeseries` to evaluate
the fluxes over a whole solution.

RHS contract
------------
* ``t``              : time [min] (used to sample the forcing)
* ``y``              : state vector, length ``dim``
* ``params``         : per-link params from the model's ``precalc``
                       (``setup.precalculations_*``)
                       -> [L, A_h, invtau, k_2, k_i, c_1, c_2]
* ``global_params``  : global parameter vector (see setup docstring)
* ``forcing``        : callable ``forcing(t) -> (rain_mm_hr, epot_mm_month)``
                       or a 2-tuple of constants
* ``parents_q``      : summed upstream channel discharge [m^3/s]
* ``parents_qb``     : summed upstream base flow         [m^3/s]

Flux dict keys (TopLayer), all hillslope rates in [m/min]
---------------------------------------------------------
    rain_in : rainfall reaching ponded storage  (rain_mm_hr * c_1)
    e_p, e_t, e_s : evaporation from ponded / top / sub-surface
    k_t     : effective infiltration coefficient        [1/min]
    q_pl    : ponded   -> channel
    q_pt    : ponded   -> top layer
    q_ts    : top      -> sub-surface
    q_sl    : sub-sfc  -> channel
    q_out   : hillslope outflow (q_pl + q_sl) * c_2   [m^3/s]
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
from scipy.integrate import solve_ivp

from setup import (
    precalculations_toplayer, precalculations_crb3, precalculations_basic,
    precalculations_constant_runoff,
    GLOBAL_PARAMS_TOPLAYER, GLOBAL_PARAMS_CRB3, GLOBAL_PARAMS_NZV1,
    GLOBAL_PARAMS_CONSTANT_RUNOFF, GLOBAL_PARAMS_XINANJIANG,
    GLOBAL_PARAMS_TANK_SMA,
)


# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class HLMModel:
    """Metadata + right-hand side (+ optional flux extractor) for one variant.

    The fields after ``fluxes`` make a model runnable by :func:`simulate` (and
    therefore by the web app, see ``app.py``); a model is "runnable" when
    ``precalc``, ``default_params``, ``y0`` and ``streamflow`` are all set.
    """
    name: str
    dydt: Callable
    dim: int
    state_names: Tuple[str, ...]
    description: str = ""
    fluxes: Optional[Callable] = None     # fluxes(t, y, params, gp, forcing) -> dict
    precalc: Optional[Callable] = None    # precalc(L, A_h, A_i, gp) -> params
    param_names: Tuple[Tuple[str, str], ...] = ()   # (name, unit) per gp index
    default_params: Optional[Tuple[float, ...]] = None
    y0: Optional[Tuple[float, ...]] = None          # default initial state
    storages: Tuple[str, ...] = ()        # state names that are storages [m]
    streamflow: Optional[str] = None      # state name or flux key, [m^3/s]
    # Equations, for display (the app's "Equations" view):
    #   balance   : {storage: (inflow flux keys, outflow flux keys)}
    #   flux_eqs  : {flux key: (LaTeX symbol, LaTeX right-hand side)}
    #   extra_eqs : ((LaTeX lhs, LaTeX rhs), ...) definitions / routing
    balance: Optional[dict] = None
    flux_eqs: Optional[dict] = None
    extra_eqs: Tuple[Tuple[str, str], ...] = ()
    #   diagram   : {flux key: (from, to) or None}, overrides of flow_edges()
    diagram: Optional[dict] = None

    @property
    def runnable(self):
        return None not in (self.precalc, self.default_params, self.y0,
                            self.streamflow)


MODELS = {}


def register_model(name, dim, state_names, description="", fluxes=None,
                   precalc=None, param_names=(), default_params=None, y0=None,
                   storages=(), streamflow=None, balance=None,
                   flux_eqs=None, extra_eqs=(), diagram=None):
    """Decorator: register a dydt function as a named model.

    ``fluxes`` is an optional companion function returning a dict of internal
    fluxes for the same state; if given, :func:`flux_timeseries` can reconstruct
    the fluxes over a solution. The remaining keywords are documented on
    :class:`HLMModel`.
    """
    if default_params is not None and len(param_names) != len(default_params):
        raise ValueError(f"{name}: param_names and default_params differ in length")
    if y0 is not None and len(y0) != dim:
        raise ValueError(f"{name}: y0 must have length dim={dim}")

    def deco(fn):
        MODELS[name] = HLMModel(
            name=name, dydt=fn, dim=dim,
            state_names=tuple(state_names), description=description,
            fluxes=fluxes, precalc=precalc,
            param_names=tuple(param_names),
            default_params=None if default_params is None
                           else tuple(float(v) for v in default_params),
            y0=None if y0 is None else tuple(float(v) for v in y0),
            storages=tuple(storages), streamflow=streamflow,
            balance=balance, flux_eqs=flux_eqs, extra_eqs=tuple(extra_eqs),
            diagram=diagram,
        )
        return fn
    return deco


# Source / sink nodes of the flow diagram.
RAIN, ET, CHANNEL = "Rain", "ET", "Channel"


def flow_edges(model):
    """Flow-diagram edges ``[(flux_key, from, to), ...]`` derived from ``balance``.

    A flux leaving one storage and entering another links the two. A flux that
    only enters storages comes from ``RAIN``; one that only leaves goes to
    ``ET`` if its key starts with ``e`` (evaporation), else to ``CHANNEL``.
    ``model.diagram`` overrides single fluxes (``None`` hides one), for flows
    the balance alone cannot place, such as a flux split between two targets.
    """
    src, dst = {}, {}
    for storage, (ins, outs) in model.balance.items():
        for k in ins:
            dst[k] = storage
        for k in outs:
            src[k] = storage
    edges = {}
    for k in list(src) + [k for k in dst if k not in src]:
        default_sink = ET if k.startswith("e") else CHANNEL
        edges[k] = (src.get(k, RAIN), dst.get(k, default_sink))
    edges.update(model.diagram or {})
    return [(k, a, b) for k, ab in edges.items() if ab is not None
            for a, b in [ab]]


def get_model(name):
    """Look up a registered model by name (raises if unknown)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {list(MODELS)}")
    return MODELS[name]


def flux_timeseries(model, t, y, params, global_params, forcing):
    """Evaluate a model's fluxes at every time point of a solution.

    Parameters
    ----------
    model : HLMModel
        A model with a non-None ``fluxes`` companion.
    t : array_like, shape (N,)
        Times [min] (e.g. ``sol.t``).
    y : array_like, shape (dim, N)
        States, one column per time (e.g. ``sol.y``).
    params, global_params, forcing :
        Same objects passed to the integration.

    Returns
    -------
    dict[str, numpy.ndarray]
        One array of shape (N,) per flux key. The set of keys is whatever the
        model's ``fluxes`` function returns.
    """
    if model.fluxes is None:
        raise ValueError(f"model {model.name!r} has no flux function registered")
    t = np.atleast_1d(np.asarray(t, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if y.shape[0] != model.dim and y.shape[1] == model.dim:
        y = y.T                                   # accept (N, dim) too
    n = t.size
    out = {}
    for k in range(n):
        f = model.fluxes(t[k], y[:, k], params, global_params, forcing)
        for key, val in f.items():
            if key not in out:
                out[key] = np.empty(n)
            out[key][k] = val
    return out


def simulate(model, L, A_h, A_i, forcing, t_end, global_params=None, y0=None,
             t_eval=None):
    """Run one single-link simulation of a runnable model.

    Parameters
    ----------
    model : HLMModel
    L, A_h, A_i : channel length [m], hillslope area [m^2], upstream area [km^2]
    forcing : ArrayForcing or 2-tuple (see RHS contract)
    t_end : float, simulation length [min]
    global_params, y0 : override the model's defaults if given
    t_eval : times [min] at which to store the solution

    Returns
    -------
    dict with ``t`` [min], ``y`` (dim, N), ``fluxes`` (dict of arrays) and
    ``q`` (streamflow [m^3/s], shape (N,)).
    """
    if not model.runnable:
        raise ValueError(f"model {model.name!r} is missing simulation metadata")
    gp = np.array(model.default_params if global_params is None else global_params,
                  dtype=float)
    y0 = np.array(model.y0 if y0 is None else y0, dtype=float)
    params = model.precalc(L, A_h, A_i, gp)

    sol = solve_ivp(model.dydt, (0.0, t_end), y0, args=(params, gp, forcing),
                    method="LSODA", rtol=1e-6, atol=1e-9, max_step=30.0,
                    t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"{model.name}: {sol.message}")

    f = flux_timeseries(model, sol.t, sol.y, params, gp, forcing)
    if model.streamflow in model.state_names:
        q = sol.y[model.state_names.index(model.streamflow)]
    else:
        q = f[model.streamflow]
    return {"t": sol.t, "y": sol.y, "fluxes": f, "q": q}


def route_channel(t, q_in, v_0, lambda_1, lambda_2, L, A_i, q0=None):
    """Route a hillslope outflow through the HLM non-linear channel reservoir.

    Solves asynch's channel equation
    ``dq/dt = invtau * q**lambda_1 * (-q + q_in(t))`` with
    ``invtau = 60 v_0 A_i**lambda_2 / ((1 - lambda_1) L)``.

    Parameters
    ----------
    t : array_like, shape (N,)
        Times [min]; ``q_in`` is interpolated linearly between them.
    q_in : array_like, shape (N,)
        Inflow to the channel [m^3/s], e.g. a model's ``q_out``.
    v_0 : float, channel reference velocity [m/s]
    lambda_1, lambda_2 : float, discharge and area exponents [-]
    L : float, channel length the water travels [m]
    A_i : float, upstream area [km^2]
    q0 : float, initial discharge [m^3/s]; defaults to ``q_in[0]``
        (steady state), floored at 1e-6 so ``q**lambda_1`` is not stuck at 0.

    Returns
    -------
    numpy.ndarray, shape (N,) : routed discharge [m^3/s] at ``t``.
    """
    t = np.asarray(t, dtype=float)
    q_in = np.asarray(q_in, dtype=float)
    invtau = 60.0 * v_0 * A_i ** lambda_2 / ((1.0 - lambda_1) * L)   # [1/min]
    q0 = max(q_in[0] if q0 is None else q0, 1e-6)

    def rhs(tt, q):
        qq = max(q[0], 1e-9)
        return [invtau * qq ** lambda_1 * (-qq + np.interp(tt, t, q_in))]

    sol = solve_ivp(rhs, (t[0], t[-1]), [q0], t_eval=t, method="LSODA",
                    rtol=1e-6, atol=1e-9, max_step=10.0)
    if not sol.success:
        raise RuntimeError(f"channel routing: {sol.message}")
    return sol.y[0]


def _read_forcing(forcing, t):
    """Sample forcing at time t and convert e_pot to [m/min]."""
    rain_mm_hr, epot_mm_month = forcing(t) if callable(forcing) else forcing
    e_pot = epot_mm_month * (1e-3 / (30.0 * 24.0 * 60.0))   # mm/month -> m/min
    return rain_mm_hr, e_pot


def _evaporation(s_p, s_t, s_s, e_pot, S_L, h_b):
    """Evaporation partitioning of the TopLayer model (asynch 254/255)."""
    Corr = s_p + s_t / S_L + s_s / (h_b - S_L)
    if e_pot > 0.0 and Corr > 1e-12:
        e_p = s_p * e_pot / Corr
        e_t = s_t / S_L * e_pot / Corr
        e_s = s_s / (h_b - S_L) * e_pot / Corr
        return e_p, e_t, e_s
    return 0.0, 0.0, 0.0


# ----------------------------------------------------------------------------
# TopLayer -- asynch 254/255 hillslope: ponded / top layer / sub-surface
# ----------------------------------------------------------------------------
# Hillslope part of asynch model 255 (254 with a non-linear sub-surface ->
# channel release). The channel (q) and baseflow (q_b) routing states and
# their parameters (v_0, lambda_1, lambda_2, v_B) are not part of this model.
def fluxes_toplayer(t, y, params, global_params, forcing):
    k_3      = global_params[1]     # [1/min]
    h_b      = global_params[3]     # [m]
    S_L      = global_params[4]     # [m]
    A        = global_params[5]
    B        = global_params[6]
    exponent = global_params[7]
    Ss_max   = global_params[8]     # [m]
    Ss_min   = global_params[9]     # [m]
    Ss_coef  = global_params[10]    # [1/min]
    Ss_exp   = global_params[11]    # [1/m]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_p, s_t, s_s = y[0], y[1], y[2]

    e_p, e_t, e_s = _evaporation(s_p, s_t, s_s, e_pot, S_L, h_b)

    # saturation-excess infiltration into the top layer
    pow_term = (1.0 - s_t / S_L) ** exponent if (1.0 - s_t / S_L) > 0.0 else 0.0
    k_t = (A + B * pow_term) * k_2

    # non-linear sub-surface -> channel release
    q_sl = 0.0
    if s_s > Ss_min:
        q_sl = k_3 * s_s
    if s_s > Ss_max:
        q_sl += Ss_coef * (s_s - Ss_max) * np.exp(Ss_exp * (s_s - Ss_max))

    q_pl = k_2 * s_p
    return {
        "rain_in": rain_mm_hr * c_1,
        "e_p": e_p, "e_t": e_t, "e_s": e_s,
        "k_t": k_t,
        "q_pl": q_pl,               # ponded  -> channel
        "q_pt": k_t * s_p,          # ponded  -> top
        "q_ts": k_i * s_t,          # top     -> sub-surface
        "q_sl": q_sl,               # sub-sfc -> channel
        "q_out": (q_pl + q_sl) * c_2,               # hillslope outflow [m^3/s]
    }


@register_model(
    "TopLayer", dim=3,
    state_names=("s_p", "s_t", "s_s"),
    description="Top Layer (asynch 254/255 hillslope): ponded / top-layer / "
                "sub-surface storages with saturation-excess infiltration and "
                "a non-linear sub-surface release. Streamflow is the 'q_out' "
                "flux [m^3/s].",
    fluxes=fluxes_toplayer,
    precalc=precalculations_toplayer,
    param_names=(("v_h", "m/s"), ("k_3", "1/min"), ("k_I_factor", "-"),
                 ("h_b", "m"), ("S_L", "m"), ("A", "-"), ("B", "-"),
                 ("exponent", "-"), ("Ssmax", "m"), ("Ssmin", "m"),
                 ("Ss_coef", "1/min"), ("Ss_exp", "1/m")),
    default_params=GLOBAL_PARAMS_TOPLAYER,
    y0=(0.0, 0.05, 0.20),
    storages=("s_p", "s_t", "s_s"), streamflow="q_out",
    balance={
        "s_p": (("rain_in",), ("q_pl", "q_pt", "e_p")),
        "s_t": (("q_pt",),    ("q_ts", "e_t")),
        "s_s": (("q_ts",),    ("q_sl", "e_s")),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "e_p":     (r"E_p",     r"\frac{S_p}{C}\,E_{pot}"),
        "e_t":     (r"E_t",     r"\frac{S_t / S_L}{C}\,E_{pot}"),
        "e_s":     (r"E_s",     r"\frac{S_s / (h_b - S_L)}{C}\,E_{pot}"),
        "k_t":     (r"k_t",     r"k_2\left(A + B\left(1 - \frac{S_t}{S_L}\right)^{\text{exponent}}\right)"),
        "q_pl":    (r"q_{pl}",  r"k_2\,S_p"),
        "q_pt":    (r"q_{pt}",  r"k_t\,S_p"),
        "q_ts":    (r"q_{ts}",  r"k_i\,S_t"),
        "q_sl":    (r"q_{sl}",  r"""\begin{cases}
            0 & S_s \le S_{s,min} \\
            k_3\,S_s & S_{s,min} < S_s \le S_{s,max} \\
            k_3\,S_s + c_s\,(S_s - S_{s,max})\,e^{\eta_s (S_s - S_{s,max})} & S_s > S_{s,max}
            \end{cases}"""),
        "q_out":   (r"Q_{out}", r"c_2\,(q_{pl} + q_{sl})"),
    },
    extra_eqs=(
        (r"C",            r"S_p + \frac{S_t}{S_L} + \frac{S_s}{h_b - S_L}"),
        (r"k_2",          r"60\,v_h\,\frac{L}{A_h}"),
        (r"k_i",          r"k_{I,\text{factor}}\;k_2"),
        (r"c_s,\ \eta_s", r"\text{Ss\_coef},\ \text{Ss\_exp}"),
        (r"c_1,\ c_2",    r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
)
def model_toplayer(t, y, params, global_params, forcing,
                   parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: routing is not part of this model.
    f = fluxes_toplayer(t, y, params, global_params, forcing)
    ans = np.zeros(3)
    ans[0] = f["rain_in"] - f["q_pl"] - f["q_pt"] - f["e_p"]     # s_p
    ans[1] = f["q_pt"] - f["q_ts"] - f["e_t"]                    # s_t
    ans[2] = f["q_ts"] - f["q_sl"] - f["e_s"]                    # s_s
    return ans


# ----------------------------------------------------------------------------
# Collie River Basin 3 (MARRMoT-style) -- soil / ponded / groundwater storages
# ----------------------------------------------------------------------------
# Smoothing of the saturation-excess threshold (MARRMoT smoothThreshStorage).
CRB3_SMOOTH_R = 0.01
CRB3_SMOOTH_E = 5.0


SMOOTH_EXCESS_EQ = (
    r"\sigma(S, S_{max})",
    r"\left[1 + \exp\left(-\frac{S - S_{max} + r\,e\,S_{max}}{r\,S_{max}}\right)\right]^{-1}"
    rf",\quad r = {CRB3_SMOOTH_R:g},\ e = {CRB3_SMOOTH_E:g}",
)


def _smooth_excess(S, Smax, r=CRB3_SMOOTH_R, e=CRB3_SMOOTH_E):
    """Fraction (0..1) of input that becomes excess as S approaches Smax."""
    x = (S - Smax + r * e * Smax) / (r * Smax)
    x = min(max(x, -50.0), 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def fluxes_crb3(t, y, params, global_params, forcing):
    A      = global_params[0]
    B      = global_params[1]
    lam    = global_params[2]
    Ssmax  = global_params[3]       # [m]
    Sfc    = global_params[4] * Ssmax   # fraction -> [m]
    M      = global_params[5]
    A2     = global_params[6]
    B2     = global_params[7]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_s, s_p, s_g = max(y[1], 0.0), max(y[2], 0.0), max(y[3], 0.0)

    P = rain_mm_hr * c_1

    # evaporation: bare soil and vegetation
    e_b = s_s / Ssmax * (1.0 - M) * e_pot
    e_v = M * e_pot if s_s > Sfc else s_s / Sfc * M * e_pot

    # saturation excess -> ponded storage
    q_sp = P * _smooth_excess(s_s, Ssmax)

    # drainage above field capacity, split to groundwater / channel
    q_ss = A * (s_s - Sfc) ** B if s_s > Sfc else 0.0
    q_sg = lam * q_ss
    q_sL = (1.0 - lam) * q_ss

    q_pL = k_2 * s_p
    q_gL = A2 * s_g ** B2

    return {
        "rain_in": P,
        "e_b": e_b, "e_v": e_v,
        "q_sp": q_sp, "q_ss": q_ss, "q_sg": q_sg, "q_sL": q_sL,
        "q_pL": q_pL, "q_gL": q_gL,
        "q_out": (q_pL + q_gL + q_sL) * c_2,    # hillslope outflow [m^3/s]
    }


@register_model(
    "CRB3", dim=4,
    state_names=("Q", "s_s", "s_p", "s_g"),
    description="Collie River Basin 3: soil (Ss), ponded (Sp) and groundwater "
                "(Sg) storages. Q is cumulative runoff [m]; streamflow is the "
                "'q_out' flux [m^3/s].",
    fluxes=fluxes_crb3,
    precalc=precalculations_crb3,
    param_names=(("A", "m^(1-B)/min"), ("B", "-"), ("lambda", "-"),
                 ("Ssmax", "m"), ("Sfc", "-"), ("M", "-"),
                 ("A2", "m^(1-B2)/min"), ("B2", "-"), ("v_h", "m/s")),
    # setup defaults with the notebook's non-stiff A and B2
    default_params=np.r_[1e-3, GLOBAL_PARAMS_CRB3[1:7], 1.0, GLOBAL_PARAMS_CRB3[8]],
    y0=(0.0, 0.05, 0.0, 0.001),
    storages=("s_s", "s_p", "s_g"), streamflow="q_out",
    balance={
        "s_s": (("rain_in",), ("e_v", "e_b", "q_sp", "q_ss")),
        "s_p": (("q_sp",),    ("q_pL",)),
        "s_g": (("q_sg",),    ("q_gL",)),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "e_v":     (r"E_v",     r"\begin{cases} M\,E_{pot} & S_s > S_{fc} \\ \frac{S_s}{S_{fc}}\,M\,E_{pot} & \text{otherwise} \end{cases}"),
        "e_b":     (r"E_b",     r"\frac{S_s}{S_{s,max}}\,(1 - M)\,E_{pot}"),
        "q_sp":    (r"q_{sp}",  r"P\,\sigma(S_s, S_{s,max})"),
        "q_ss":    (r"q_{ss}",  r"\begin{cases} A\,(S_s - S_{fc})^{B} & S_s > S_{fc} \\ 0 & \text{otherwise} \end{cases}"),
        "q_sg":    (r"q_{sg}",  r"\lambda\,q_{ss}"),
        "q_sL":    (r"q_{sL}",  r"(1 - \lambda)\,q_{ss}"),
        "q_pL":    (r"q_{pL}",  r"k_2\,S_p"),
        "q_gL":    (r"q_{gL}",  r"A_2\,S_g^{B_2}"),
        "q_out":   (r"Q_{out}", r"c_2\,(q_{pL} + q_{gL} + q_{sL})"),
    },
    extra_eqs=(
        (r"S_{fc}", r"\text{Sfc}\cdot S_{s,max}"),
        SMOOTH_EXCESS_EQ,
        (r"k_2", r"60\,v_h\,\frac{L}{A_h}"),
        (r"\frac{dQ}{dt}", r"q_{pL} + q_{gL} + q_{sL}\quad \text{(cumulative runoff)}"),
        (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
    # q_ss leaves Ss but is split: q_sg -> Sg and q_sL -> channel.
    diagram={"q_ss": None, "q_sg": ("s_s", "s_g"), "q_sL": ("s_s", CHANNEL)},
)
def model_crb3(t, y, params, global_params, forcing,
               parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: Q is not routed in this model.
    f = fluxes_crb3(t, y, params, global_params, forcing)
    ans = np.zeros(4)
    ans[0] = f["q_pL"] + f["q_gL"] + f["q_sL"]                          # Q
    ans[1] = f["rain_in"] - f["e_v"] - f["e_b"] - f["q_sp"] - f["q_ss"]  # Ss
    ans[2] = f["q_sp"] - f["q_pL"]                                      # Sp
    ans[3] = f["q_sg"] - f["q_gL"]                                      # Sg
    return ans


# ----------------------------------------------------------------------------
# New Zealand V1 -- soil-matrix (Sm) / ponded (Sp) storages
# ----------------------------------------------------------------------------
def fluxes_nzv1(t, y, params, global_params, forcing):
    S_fc  = global_params[0]        # [m]
    S_max = global_params[1]        # [m]
    V_p   = global_params[2]        # [m/s]
    V_s   = global_params[3]        # [m/s]
    V_b   = global_params[4]        # [m/s]
    a_p   = global_params[5]
    a_s   = global_params[6]
    a_b   = global_params[7]
    M     = global_params[8]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_m, s_p = max(y[1], 0.0), max(y[2], 0.0)

    P = rain_mm_hr * c_1

    # release coefficients from hillslope geometry
    K_p = L / A_h * V_p * 60.0
    K_s = L / A_h * V_s * 60.0
    K_b = L / A_h * V_b * 60.0

    # evaporation: vegetation and bare soil
    e_v = M * e_pot if s_m > S_fc else s_m / S_fc * M * e_pot
    e_b = s_m / S_max * (1.0 - M) * e_pot

    # saturation excess -> ponded storage
    q_mp = P * _smooth_excess(s_m, S_max)

    q_pL = K_p * s_p ** a_p
    q_sL = K_s * (s_m - S_fc) ** a_s if s_m > S_fc else 0.0
    q_bL = K_b * s_m ** a_b

    return {
        "rain_in": P,
        "e_v": e_v, "e_b": e_b,
        "q_mp": q_mp, "q_pL": q_pL, "q_sL": q_sL, "q_bL": q_bL,
        "q_out": (q_pL + q_sL + q_bL) * c_2,    # hillslope outflow [m^3/s]
    }


@register_model(
    "NewZealandV1", dim=3,
    state_names=("Q", "s_m", "s_p"),
    description="New Zealand V1: soil-matrix (Sm) and ponded (Sp) storages. "
                "Q is cumulative runoff [m]; streamflow is the 'q_out' flux "
                "[m^3/s].",
    fluxes=fluxes_nzv1,
    precalc=precalculations_basic,
    param_names=(("S_fc", "m"), ("S_max", "m"), ("V_p", "m/s"),
                 ("V_s", "m/s"), ("V_b", "m/s"), ("a_p", "-"), ("a_s", "-"),
                 ("a_b", "-"), ("M", "-")),
    default_params=GLOBAL_PARAMS_NZV1, y0=(0.0, 0.05, 0.0),
    storages=("s_m", "s_p"), streamflow="q_out",
    balance={
        "s_m": (("rain_in",), ("e_v", "e_b", "q_mp", "q_sL", "q_bL")),
        "s_p": (("q_mp",),    ("q_pL",)),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "e_v":     (r"E_v",     r"\begin{cases} M\,E_{pot} & S_m > S_{fc} \\ \frac{S_m}{S_{fc}}\,M\,E_{pot} & \text{otherwise} \end{cases}"),
        "e_b":     (r"E_b",     r"\frac{S_m}{S_{max}}\,(1 - M)\,E_{pot}"),
        "q_mp":    (r"q_{mp}",  r"P\,\sigma(S_m, S_{max})"),
        "q_pL":    (r"q_{pL}",  r"K_p\,S_p^{a_p}"),
        "q_sL":    (r"q_{sL}",  r"\begin{cases} K_s\,(S_m - S_{fc})^{a_s} & S_m > S_{fc} \\ 0 & \text{otherwise} \end{cases}"),
        "q_bL":    (r"q_{bL}",  r"K_b\,S_m^{a_b}"),
        "q_out":   (r"Q_{out}", r"c_2\,(q_{pL} + q_{sL} + q_{bL})"),
    },
    extra_eqs=(
        (r"K_p,\ K_s,\ K_b", r"60\,\frac{L}{A_h}\,V_p,\quad 60\,\frac{L}{A_h}\,V_s,\quad 60\,\frac{L}{A_h}\,V_b"),
        SMOOTH_EXCESS_EQ,
        (r"\frac{dQ}{dt}", r"q_{pL} + q_{sL} + q_{bL}\quad \text{(cumulative runoff)}"),
        (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
)
def model_nzv1(t, y, params, global_params, forcing,
               parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: Q is not routed in this model.
    f = fluxes_nzv1(t, y, params, global_params, forcing)
    ans = np.zeros(3)
    ans[0] = f["q_pL"] + f["q_sL"] + f["q_bL"]                          # Q
    ans[1] = (f["rain_in"] - f["e_v"] - f["e_b"] - f["q_mp"]
              - f["q_sL"] - f["q_bL"])                                  # Sm
    ans[2] = f["q_mp"] - f["q_pL"]                                      # Sp
    return ans


# ----------------------------------------------------------------------------
# ConstantRunoff -- asynch 190 hillslope (UMN_HLM_Coupled model_ConstantRunoff)
# ----------------------------------------------------------------------------
# Port of runoff/src/models/model_ConstantRunoff.hpp. The C++ model computes
# PET from t2m with Oudin; here PET comes from the forcing like every other
# playground model.
def fluxes_constant_runoff(t, y, params, global_params, forcing):
    rc  = global_params[0]
    v_g = global_params[2]          # [m/s]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_p, s_a = max(y[0], 0.0), max(y[1], 0.0)

    P = rain_mm_hr * c_1
    k_3 = v_g * L / A_h * 60.0      # baseflow rate [1/min]

    # asynch 190: PET shared in proportion to storage
    e_p = e_a = 0.0
    if e_pot > 0.0:
        C_p = s_p / e_pot
        C_a = s_a / e_pot
        C_T = C_p + C_a
        corr = 1.0 / C_T if C_T > 1.0 else 1.0
        e_p = corr * C_p * e_pot
        e_a = corr * C_a * e_pot

    q_pl = k_2 * s_p
    q_al = k_3 * s_a

    return {
        "rain_in": P,
        "q_rp": rc * P, "q_ra": (1.0 - rc) * P,     # rain -> runoff / baseflow
        "e_p": e_p, "e_a": e_a,
        "q_pl": q_pl, "q_al": q_al,
        "q_out": (q_pl + q_al) * c_2,               # hillslope outflow [m^3/s]
    }


@register_model(
    "ConstantRunoff", dim=4,
    state_names=("s_p", "s_a", "surf_runoff", "total_runoff"),
    description="Constant runoff coefficient (asynch 190 hillslope): rain is "
                "split by rc between a fast runoff (s_p) and a slow baseflow "
                "(s_a) linear reservoir. Streamflow is the 'q_out' flux "
                "[m^3/s].",
    fluxes=fluxes_constant_runoff,
    precalc=precalculations_constant_runoff,
    param_names=(("rc", "-"), ("v_h", "m/s"), ("v_g", "m/s")),
    default_params=GLOBAL_PARAMS_CONSTANT_RUNOFF,
    y0=(0.0, 0.01, 0.0, 0.0),
    storages=("s_p", "s_a"), streamflow="q_out",
    balance={
        "s_p": (("q_rp",), ("q_pl", "e_p")),
        "s_a": (("q_ra",), ("q_al", "e_a")),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "q_rp":    (r"P_p",     r"r_c\,P"),
        "q_ra":    (r"P_a",     r"(1 - r_c)\,P"),
        "e_p":     (r"E_p",     r"S_p\,\min\left(1,\ \frac{E_{pot}}{S_p + S_a}\right)"),
        "e_a":     (r"E_a",     r"S_a\,\min\left(1,\ \frac{E_{pot}}{S_p + S_a}\right)"),
        "q_pl":    (r"q_{pl}",  r"k_2\,S_p"),
        "q_al":    (r"q_{al}",  r"k_3\,S_a"),
        "q_out":   (r"Q_{out}", r"c_2\,(q_{pl} + q_{al})"),
    },
    extra_eqs=(
        (r"k_2,\ k_3", r"60\,v_h\,\frac{L}{A_h},\quad 60\,v_g\,\frac{L}{A_h}"),
        (r"\frac{d\,\text{surf\_runoff}}{dt}", r"q_{pl}\quad \text{(cumulative)}"),
        (r"\frac{d\,\text{total\_runoff}}{dt}", r"q_{pl} + q_{al}\quad \text{(cumulative)}"),
        (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
)
def model_constant_runoff(t, y, params, global_params, forcing,
                          parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: routing is not part of this model.
    f = fluxes_constant_runoff(t, y, params, global_params, forcing)
    ans = np.zeros(4)
    ans[0] = f["q_rp"] - f["q_pl"] - f["e_p"]          # s_p
    ans[1] = f["q_ra"] - f["q_al"] - f["e_a"]          # s_a
    ans[2] = f["q_pl"]                                 # surf_runoff
    ans[3] = f["q_pl"] + f["q_al"]                     # total_runoff
    return ans


# ----------------------------------------------------------------------------
# Xinanjiang (MARRMoT m_28_xinanjiang_12p_4s) + ponded storage Sp
# ----------------------------------------------------------------------------
# Port of step_model_28 in FindBestModel/marrmot_models.py. MARRMoT's surface
# runoff QS = RS + RB is instantaneous; here it fills a ponded storage Sp that
# drains as q_pL = K_p * Sp^a_1. Rates kI, kG, cI, cG are given in 1/d.
PER_DAY = 1.0 / 1440.0              # 1/d -> 1/min


def fluxes_xinanjiang(t, y, params, global_params, forcing):
    Aim, a, b, Wmax, LM, c, Smax, Ex = global_params[0:8]
    kI, kG, cI, cG = np.asarray(global_params[8:12]) * PER_DAY   # [1/min]
    v_h, a_1 = global_params[12], global_params[13]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_w, s_f, s_i, s_g, s_p = (max(v, 0.0) for v in y[0:5])

    P = rain_mm_hr * c_1
    K_p = L / A_h * v_h * 60.0

    # tension water: rain on pervious area, double-parabolic runoff curve
    P_i = (1.0 - Aim) * P
    w = min(s_w / Wmax, 1.0)
    if w <= 0.5 - a:
        R = P_i * (0.5 - a) ** (1.0 - b) * w ** b
    else:
        R = P_i * (1.0 - (0.5 + a) ** (1.0 - b) * (1.0 - w) ** b)

    if s_w > LM:
        E = e_pot
    elif s_w >= c * LM:
        E = s_w / LM * e_pot
    elif s_w > 0.0:
        E = c * e_pot
    else:
        E = 0.0                     # guard: no evaporation from an empty store

    # free water: variable contributing area
    contrib = 1.0 - (1.0 - min(s_f / Smax, 1.0)) ** Ex
    R_S = R * contrib
    R_I = kI * s_f * contrib
    R_G = kG * s_f * contrib

    R_B = Aim * P                   # impervious runoff
    q_pL = K_p * s_p ** a_1
    Q_I = cI * s_i
    Q_G = cG * s_g

    return {
        "rain_in": P,
        "p_i": P_i, "r_b": R_B,
        "e_w": E, "r_w": R,
        "r_s": R_S, "r_i": R_I, "r_g": R_G,
        "q_pL": q_pL, "q_i": Q_I, "q_g": Q_G,
        "q_out": (q_pL + Q_I + Q_G) * c_2,      # hillslope outflow [m^3/s]
    }


@register_model(
    "Xinanjiang", dim=5,
    state_names=("s_w", "s_f", "s_i", "s_g", "s_p"),
    description="Xinanjiang (MARRMoT 28) with a ponded storage: tension water "
                "(s_w), free water (s_f), interflow (s_i) and baseflow (s_g) "
                "reservoirs; surface runoff RS + RB fills s_p, drained as "
                "K_p*Sp^a_1. Streamflow is the 'q_out' flux [m^3/s].",
    fluxes=fluxes_xinanjiang,
    precalc=precalculations_basic,
    param_names=(("Aim", "-"), ("a", "-"), ("b", "-"), ("Wmax", "m"),
                 ("LM", "m"), ("c", "-"), ("Smax", "m"), ("Ex", "-"),
                 ("kI", "1/d"), ("kG", "1/d"), ("cI", "1/d"), ("cG", "1/d"),
                 ("v_h", "m/s"), ("a_1", "-")),
    default_params=GLOBAL_PARAMS_XINANJIANG,
    y0=(0.05, 0.005, 0.001, 0.005, 0.0),
    storages=("s_w", "s_f", "s_i", "s_g", "s_p"), streamflow="q_out",
    balance={
        "s_w": (("p_i",),        ("e_w", "r_w")),
        "s_f": (("r_w",),        ("r_s", "r_i", "r_g")),
        "s_i": (("r_i",),        ("q_i",)),
        "s_g": (("r_g",),        ("q_g",)),
        "s_p": (("r_b", "r_s"),  ("q_pL",)),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "p_i":     (r"P_i",     r"(1 - A_{im})\,P"),
        "r_b":     (r"R_B",     r"A_{im}\,P"),
        "e_w":     (r"E",       r"\begin{cases} E_{pot} & S_w > L_M \\ \frac{S_w}{L_M}\,E_{pot} & c\,L_M \le S_w \le L_M \\ c\,E_{pot} & \text{otherwise} \end{cases}"),
        "r_w":     (r"R",       r"\begin{cases} P_i\,(0.5 - a)^{1-b}\,\left(\frac{S_w}{W_{max}}\right)^{b} & \frac{S_w}{W_{max}} \le 0.5 - a \\ P_i\left[1 - (0.5 + a)^{1-b}\left(1 - \frac{S_w}{W_{max}}\right)^{b}\right] & \text{otherwise} \end{cases}"),
        "r_s":     (r"R_S",     r"R\,\phi"),
        "r_i":     (r"R_I",     r"k_I\,S_f\,\phi"),
        "r_g":     (r"R_G",     r"k_G\,S_f\,\phi"),
        "q_pL":    (r"q_{pL}",  r"K_p\,S_p^{a_1}"),
        "q_i":     (r"Q_I",     r"c_I\,S_i"),
        "q_g":     (r"Q_G",     r"c_G\,S_g"),
        "q_out":   (r"Q_{out}", r"c_2\,(q_{pL} + Q_I + Q_G)"),
    },
    extra_eqs=(
        (r"\phi", r"1 - \left(1 - \frac{S_f}{S_{max}}\right)^{E_x}\quad \text{(contributing area)}"),
        (r"K_p", r"60\,v_h\,\frac{L}{A_h}"),
        (r"k_I,\ k_G,\ c_I,\ c_G", r"\text{given in 1/d, used in 1/min}\ (\div 1440)"),
        (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
)
def model_xinanjiang(t, y, params, global_params, forcing,
                     parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: routing is not part of this model.
    f = fluxes_xinanjiang(t, y, params, global_params, forcing)
    ans = np.zeros(5)
    ans[0] = f["p_i"] - f["e_w"] - f["r_w"]                      # s_w
    ans[1] = f["r_w"] - f["r_s"] - f["r_i"] - f["r_g"]           # s_f
    ans[2] = f["r_i"] - f["q_i"]                                 # s_i
    ans[3] = f["r_g"] - f["q_g"]                                 # s_g
    ans[4] = f["r_b"] + f["r_s"] - f["q_pL"]                     # s_p
    return ans


# ----------------------------------------------------------------------------
# Tank Model - SMA (MARRMoT m_38_tank2_16p_5s)
# ----------------------------------------------------------------------------
# Port of step_model_38 in FindBestModel/marrmot_models.py, written for a
# continuous ODE solver:
#   * threshold outlets use the literal A*max(S - t, 0). MARRMoT's logistic
#     smoothing switches on at 0.95*t, where S - t < 0, so it gives negative
#     outflows just below each outlet;
#   * the evaporation cascade (E2 only once S1 is empty, E3 once S1 and S2
#     are, ...) and the capillary-rise limit use the ramp _tank_on(S) =
#     min(S/eps, 1) instead of hard switches the solver would chatter on. It
#     reaches 0 exactly at an empty tank, so storages cannot go negative (as
#     MARRMoT's own min(Ep, S/dt) evaporation limit).
# Tank outlets are linear reservoirs, so no extra ponded storage is needed.
TANK_EPS = 1e-4                     # on/off ramp width [m] (= 0.1 mm)


def _tank_on(S):
    """0 when storage S [m] is empty, rising linearly to 1 at TANK_EPS."""
    return min(max(S / TANK_EPS, 0.0), 1.0)


def fluxes_tank_sma(t, y, params, global_params, forcing):
    sm1, sm2 = global_params[0], global_params[1]                    # [m]
    k1, k2 = np.asarray(global_params[2:4]) * 1e-3 * PER_DAY          # mm/d -> m/min
    A0, A1, A2 = np.asarray(global_params[4:7]) * PER_DAY             # [1/min]
    t1, t2 = global_params[7], global_params[8]                      # [m]
    B0, B1 = np.asarray(global_params[9:11]) * PER_DAY
    t3 = global_params[11]
    C0, C1 = np.asarray(global_params[12:14]) * PER_DAY
    t4 = global_params[14]
    D1 = global_params[15] * PER_DAY

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    on1, on2, on3, on4 = (_tank_on(v) for v in (y[0], y[2], y[3], y[4]))
    s_1, x_s, s_2, s_3, s_4 = (max(v, 0.0) for v in y[0:5])

    P = rain_mm_hr * c_1

    # soil moisture: capillary rise S2 -> S1 and exchange S1 <-> Xs
    T1 = k1 * (1.0 - s_1 / sm1) * on2 if s_1 < sm1 else 0.0
    T2 = k2 * (min(s_1, sm1) / sm1 - x_s / sm2)

    # evaporation from the top-most non-empty tank
    E1 = e_pot * on1
    E2 = e_pot * (1.0 - on1) * on2
    E3 = e_pot * (1.0 - on1) * (1.0 - on2) * on3
    E4 = e_pot * (1.0 - on1) * (1.0 - on2) * (1.0 - on3) * on4

    # drainage between tanks and side outlets
    F12 = A0 * max(s_1 - sm1, 0.0)
    Y1 = A1 * max(s_1 - t1, 0.0)
    Y2 = A2 * max(s_1 - t2, 0.0)
    F23 = B0 * s_2
    Y3 = B1 * max(s_2 - t3, 0.0)
    F34 = C0 * s_3
    Y4 = C1 * max(s_3 - t4, 0.0)
    Y5 = D1 * s_4

    return {
        "rain_in": P,
        "t_1": T1, "t_2": T2,
        "e_1": E1, "e_2": E2, "e_3": E3, "e_4": E4,
        "f_12": F12, "f_23": F23, "f_34": F34,
        "y_1": Y1, "y_2": Y2, "y_3": Y3, "y_4": Y4, "y_5": Y5,
        "q_out": (Y1 + Y2 + Y3 + Y4 + Y5) * c_2,   # hillslope outflow [m^3/s]
    }


@register_model(
    "TankSMA", dim=5,
    state_names=("s_1", "x_s", "s_2", "s_3", "s_4"),
    description="Tank Model - SMA (MARRMoT 38): top tank S1 with secondary "
                "soil moisture Xs, then intermediate S2, sub-base S3 and base "
                "S4 tanks, each with side outlets. Rates in 1/d (k1, k2 in "
                "mm/d). Streamflow is the 'q_out' flux [m^3/s].",
    fluxes=fluxes_tank_sma,
    precalc=precalculations_basic,
    param_names=(("sm1", "m"), ("sm2", "m"), ("k1", "mm/d"), ("k2", "mm/d"),
                 ("A0", "1/d"), ("A1", "1/d"), ("A2", "1/d"), ("t1", "m"),
                 ("t2", "m"), ("B0", "1/d"), ("B1", "1/d"), ("t3", "m"),
                 ("C0", "1/d"), ("C1", "1/d"), ("t4", "m"), ("D1", "1/d")),
    default_params=GLOBAL_PARAMS_TANK_SMA,
    y0=(0.01, 0.03, 0.01, 0.02, 0.05),
    storages=("s_1", "x_s", "s_2", "s_3", "s_4"), streamflow="q_out",
    balance={
        "s_1": (("rain_in", "t_1"), ("t_2", "e_1", "f_12", "y_1", "y_2")),
        "x_s": (("t_2",),           ()),
        "s_2": (("f_12",),          ("e_2", "t_1", "f_23", "y_3")),
        "s_3": (("f_23",),          ("e_3", "f_34", "y_4")),
        "s_4": (("f_34",),          ("e_4", "y_5")),
    },
    flux_eqs={
        "rain_in": (r"P",       r"c_1\,p(t)"),
        "t_1":     (r"T_1",     r"\begin{cases} k_1\left(1 - \frac{S_1}{sm_1}\right)\omega(S_2) & S_1 < sm_1 \\ 0 & \text{otherwise} \end{cases}"),
        "t_2":     (r"T_2",     r"k_2\left(\frac{\min(S_1, sm_1)}{sm_1} - \frac{X_s}{sm_2}\right)"),
        "e_1":     (r"E_1",     r"E_{pot}\,\omega(S_1)"),
        "e_2":     (r"E_2",     r"E_{pot}\,[1 - \omega(S_1)]\,\omega(S_2)"),
        "e_3":     (r"E_3",     r"E_{pot}\,[1 - \omega(S_1)][1 - \omega(S_2)]\,\omega(S_3)"),
        "e_4":     (r"E_4",     r"E_{pot}\,[1 - \omega(S_1)][1 - \omega(S_2)][1 - \omega(S_3)]\,\omega(S_4)"),
        "f_12":    (r"F_{12}",  r"A_0\,\max(S_1 - sm_1,\,0)"),
        "y_1":     (r"Y_1",     r"A_1\,\max(S_1 - t_1,\,0)"),
        "y_2":     (r"Y_2",     r"A_2\,\max(S_1 - t_2,\,0)"),
        "f_23":    (r"F_{23}",  r"B_0\,S_2"),
        "y_3":     (r"Y_3",     r"B_1\,\max(S_2 - t_3,\,0)"),
        "f_34":    (r"F_{34}",  r"C_0\,S_3"),
        "y_4":     (r"Y_4",     r"C_1\,\max(S_3 - t_4,\,0)"),
        "y_5":     (r"Y_5",     r"D_1\,S_4"),
        "q_out":   (r"Q_{out}", r"c_2\,(Y_1 + Y_2 + Y_3 + Y_4 + Y_5)"),
    },
    extra_eqs=(
        (r"\omega(S)", r"\min\left(\frac{S}{\varepsilon},\ 1\right),\quad \varepsilon = 10^{-4}\ \text{m}\quad \text{(0 for an empty tank)}"),
        (r"k_1, k_2;\ A, B, C, D", r"\text{given in mm/d and 1/d, used in m/min and 1/min}"),
        (r"c_1,\ c_2", r"\frac{10^{-3}}{60}\ \text{(mm/h} \to \text{m/min)},\quad \frac{A_h}{60}\ \text{(m/min} \to \text{m}^3\text{/s)}"),
    ),
)
def model_tank_sma(t, y, params, global_params, forcing,
                   parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: routing is not part of this model.
    f = fluxes_tank_sma(t, y, params, global_params, forcing)
    ans = np.zeros(5)
    ans[0] = (f["rain_in"] + f["t_1"] - f["t_2"] - f["e_1"] - f["f_12"]
              - f["y_1"] - f["y_2"])                                     # s_1
    ans[1] = f["t_2"]                                                    # x_s
    ans[2] = f["f_12"] - f["e_2"] - f["t_1"] - f["f_23"] - f["y_3"]      # s_2
    ans[3] = f["f_23"] - f["e_3"] - f["f_34"] - f["y_4"]                 # s_3
    ans[4] = f["f_34"] - f["e_4"] - f["y_5"]                             # s_4
    return ans
