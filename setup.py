"""Shared setup helpers for HLM testing.

Contains the things that are *independent of the specific model RHS*:

* ``precalculations_*`` -- per-link parameter derivation for each model
  (``k_2, k_i, c_1, c_2``, ...) from the link geometry ``L, A_h``.
* :class:`ArrayForcing`   -- piecewise-constant forcing driven by per-record
  arrays (e.g. one record per hour).
* ``GLOBAL_PARAMS_<MODEL>`` -- default global-parameter vectors.

The model right-hand sides themselves live in ``models.py``.

Global-parameter vector layout (TopLayer, ``GLOBAL_PARAMS_TOPLAYER``)
-------------------------------------------------------------------
Hillslope of asynch model 255 (no channel / baseflow routing):
    [0]  v_h          hillslope velocity (k_2)     [m/s]
    [1]  k_3          sub-surface -> channel rate  [1/min]
    [2]  k_I_factor   k_i = k_I_factor * k_2       [-]
    [3]  h_b          total active soil depth      [m]
    [4]  S_L          top-layer depth              [m]
    [5]  A            infiltration, k_t = k_2 (A + B (1 - s_t/S_L)^exponent)
    [6]  B
    [7]  exponent
    [8]  Ssmax        non-linear release threshold [m]
    [9]  Ssmin        no release below             [m]
    [10] Ss_coef      non-linear release coef      [1/min]
    [11] Ss_exp       non-linear release exponent  [1/m]
Per-link params come from :func:`precalculations_toplayer`.

Global-parameter vector layout (Collie River Basin 3, ``GLOBAL_PARAMS_CRB3``)
---------------------------------------------------------------------------
    [0]  A        coef qss = A*(Ss - Sfc*Ssmax)^B    [m^(1-B)/min]
    [1]  B        exp qss                            [-]
    [2]  lambda   fraction of qss going to Sg        [-]
    [3]  Ssmax    max soil storage                   [m]
    [4]  Sfc      field capacity as fraction of Ssmax [-]
    [5]  M        forest fraction                    [-]
    [6]  A2       coef qgL = A2*Sg^B2                [m^(1-B2)/min]
    [7]  B2       exp qgL                            [-]
    [8]  v_h      hillslope velocity (for k_2)       [m/s]
Per-link params come from :func:`precalculations_crb3`.

Global-parameter vector layout (New Zealand V1, ``GLOBAL_PARAMS_NZV1``)
-----------------------------------------------------------------------
    [0]  S_fc     field capacity                     [m]
    [1]  S_max    max soil-matrix storage            [m]
    [2]  V_p      ponded velocity, K_p = L/A_h*V_p*60 [m/s]
    [3]  V_s      sub-surface velocity (K_s)         [m/s]
    [4]  V_b      base-flow velocity (K_b)           [m/s]
    [5]  a_p      exp qpL = K_p*Sp^a_p               [-]
    [6]  a_s      exp qsL = K_s*(Sm - S_fc)^a_s      [-]
    [7]  a_b      exp qbL = K_b*Sm^a_b               [-]
    [8]  M        forest fraction                    [-]
Per-link params come from :func:`precalculations_basic`.

Global-parameter vector layout (ConstantRunoff, ``GLOBAL_PARAMS_CONSTANT_RUNOFF``)
-------------------------------------------------------------------------------
    [0]  rc       runoff coefficient (rain -> fast)  [-]
    [1]  v_h      runoff reference velocity (k_2)    [m/s]
    [2]  v_g      baseflow reference velocity (k_3)  [m/s]
Per-link params come from :func:`precalculations_constant_runoff`.

Global-parameter vector layout (Xinanjiang + ponding, ``GLOBAL_PARAMS_XINANJIANG``)
---------------------------------------------------------------------------------
MARRMoT m_28_xinanjiang_12p_4s with depths in m and rates kept in 1/d:
    [0]  Aim      impervious fraction                [-]
    [1]  a        contributing-area curve inflection [-]
    [2]  b        contributing-area curve shape      [-]
    [3]  Wmax     max tension water storage          [m]
    [4]  LM       evaporation threshold              [m]
    [5]  c        evaporation reduction factor       [-]
    [6]  Smax     max free water storage             [m]
    [7]  Ex       free-water contributing-area shape [-]
    [8]  kI       interflow generation coefficient   [1/d]
    [9]  kG       baseflow generation coefficient    [1/d]
    [10] cI       interflow routing coefficient      [1/d]
    [11] cG       baseflow routing coefficient       [1/d]
    [12] v_h      ponded velocity, K_p = L/A_h*v_h*60 [m/s]
    [13] a_1      exp q_pL = K_p*Sp^a_1              [-]
Per-link params come from :func:`precalculations_basic`.

Global-parameter vector layout (Tank Model - SMA, ``GLOBAL_PARAMS_TANK_SMA``)
---------------------------------------------------------------------------
MARRMoT m_38_tank2_16p_5s with depths in m and rates kept in MARRMoT units:
    [0]  sm1      primary soil moisture capacity     [m]
    [1]  sm2      max secondary soil moisture (Xs)   [m]
    [2]  k1       capillary rise rate (S2 -> S1)     [mm/d]
    [3]  k2       soil moisture exchange (S1 <-> Xs) [mm/d]
    [4]  A0       drainage S1 -> S2                  [1/d]
    [5]  A1       top-tank upper outlet (Y1)         [1/d]
    [6]  A2       top-tank lower outlet (Y2)         [1/d]
    [7]  t1       Y1 outlet height                   [m]
    [8]  t2       Y2 outlet height                   [m]
    [9]  B0       drainage S2 -> S3                  [1/d]
    [10] B1       intermediate outlet (Y3)           [1/d]
    [11] t3       Y3 outlet height                   [m]
    [12] C0       drainage S3 -> S4                  [1/d]
    [13] C1       sub-base outlet (Y4)               [1/d]
    [14] t4       Y4 outlet height                   [m]
    [15] D1       base outlet (Y5)                   [1/d]
Per-link params come from :func:`precalculations_basic`.
"""

import numpy as np


# Defaults of the asynch 255 hillslope (operational model-254/255 values).
GLOBAL_PARAMS_TOPLAYER = np.array([
    0.02,        # 0  v_h        [m/s]
    2.0425e-6,   # 1  k_3        [1/min]
    0.02,        # 2  k_I_factor [-]
    0.50,        # 3  h_b        [m]
    0.10,        # 4  S_L        [m]
    0.0,         # 5  A
    99.0,        # 6  B
    3.0,         # 7  exponent
    0.20,        # 8  Ssmax      [m]
    0.05,        # 9  Ssmin      [m]
    2.0425e-6,   # 10 Ss_coef    [1/min]
    17.0,        # 11 Ss_exp     [1/m]
])

GLOBAL_PARAMS_CRB3 = np.array([
    0.1,          # 0 A       coef qss = A(Ss-Sfc)^B     [m^(1-B)/min]
    1.0,          # 1 B       exp qss                    [-]
    0.5,          # 2 lambda  fraction of qss to Sg      [-]
    0.15,         # 3 Ssmax   max soil storage           [m]
    0.20,         # 4 Sfc     field capacity / Ssmax     [-]
    0.1,          # 5 M       forest fraction            [-]
    0.003,        # 6 A2      coef qgL = A2*Sg^B2        [m^(1-B2)/min]
    0.20,         # 7 B2      exp qgL                    [-]
    0.02,         # 8 v_h     hillslope velocity (k_2)   [m/s]
])

GLOBAL_PARAMS_NZV1 = np.array([
    0.03,         # 0 S_fc    field capacity             [m]
    0.15,         # 1 S_max   max soil-matrix storage    [m]
    0.02,         # 2 V_p     ponded velocity            [m/s]
    0.002,        # 3 V_s     sub-surface velocity       [m/s]
    0.0002,       # 4 V_b     base-flow velocity         [m/s]
    1.0,          # 5 a_p     exp qpL                    [-]
    1.0,          # 6 a_s     exp qsL                    [-]
    1.0,          # 7 a_b     exp qbL                    [-]
    0.1,          # 8 M       forest fraction            [-]
])

# Defaults from UMN_HLM_Coupled (qgis_plugin runoff_models.py / Stony Brook).
GLOBAL_PARAMS_CONSTANT_RUNOFF = np.array([
    0.33,         # 0 rc      runoff coefficient         [-]
    0.1,          # 1 v_h     runoff velocity (k_2)      [m/s]
    2.2917e-5,    # 2 v_g     baseflow velocity (k_3)    [m/s]
])

GLOBAL_PARAMS_XINANJIANG = np.array([
    0.05,         # 0  Aim    impervious fraction        [-]
    0.0,          # 1  a      area-curve inflection      [-]
    0.4,          # 2  b      area-curve shape           [-]
    0.15,         # 3  Wmax   max tension water          [m]
    0.10,         # 4  LM     evaporation threshold      [m]
    0.15,         # 5  c      evaporation reduction      [-]
    0.03,         # 6  Smax   max free water             [m]
    1.5,          # 7  Ex     free-water area shape      [-]
    0.3,          # 8  kI     interflow generation       [1/d]
    0.1,          # 9  kG     baseflow generation        [1/d]
    0.5,          # 10 cI     interflow routing          [1/d]
    0.05,         # 11 cG     baseflow routing           [1/d]
    0.02,         # 12 v_h    ponded velocity (K_p)      [m/s]
    1.0,          # 13 a_1    exp q_pL                   [-]
])

GLOBAL_PARAMS_TANK_SMA = np.array([
    0.02,         # 0  sm1    primary soil moisture      [m]
    0.10,         # 1  sm2    max secondary soil moisture[m]
    1.0,          # 2  k1     capillary rise rate        [mm/d]
    1.0,          # 3  k2     soil moisture exchange     [mm/d]
    0.1,          # 4  A0     drainage S1 -> S2          [1/d]
    0.5,          # 5  A1     upper outlet (Y1)          [1/d]
    0.3,          # 6  A2     lower outlet (Y2)          [1/d]
    0.06,         # 7  t1     Y1 outlet height           [m]
    0.03,         # 8  t2     Y2 outlet height           [m]
    0.05,         # 9  B0     drainage S2 -> S3          [1/d]
    0.2,          # 10 B1     intermediate outlet (Y3)   [1/d]
    0.02,         # 11 t3     Y3 outlet height           [m]
    0.02,         # 12 C0     drainage S3 -> S4          [1/d]
    0.1,          # 13 C1     sub-base outlet (Y4)       [1/d]
    0.02,         # 14 t4     Y4 outlet height           [m]
    0.01,         # 15 D1     base outlet (Y5)           [1/d]
])


def precalculations_toplayer(L, A_h, A_i, global_params):
    """Per-link parameters for the TopLayer model (asynch 254/255 hillslope).

    Mirrors Asynch's ``Precalculations`` for the hillslope rates; there is no
    channel routing, so ``invtau`` is NaN. ``A_i`` is accepted for signature
    parity only.

    Parameters
    ----------
    L   : channel length              [m]
    A_h : hillslope area              [m^2]
    A_i : total upstream area         [km^2]   (unused)
    global_params : sequence          global parameter vector

    Returns
    -------
    numpy.ndarray, shape (7,) -> [L, A_h, nan, k_2, k_i, c_1, c_2]
    """
    v_h        = global_params[0]
    k_I_factor = global_params[2]

    k_2 = v_h * L / A_h * 60.0     # [1/min]
    k_i = k_2 * k_I_factor         # [1/min]
    c_1 = 0.001 / 60.0             # rainfall mm/hr -> m/min
    c_2 = A_h / 60.0               # m/min * A_h -> m^3/s

    return np.array([L, A_h, np.nan, k_2, k_i, c_1, c_2])


def precalculations_crb3(L, A_h, A_i, global_params):
    """Per-link parameters for the Collie River Basin 3 model.

    Same output layout as :func:`precalculations` so models share one
    ``params`` contract. CRB3 has no channel routing and no top->sub-surface
    split, so ``invtau`` and ``k_i`` are NaN (unused). ``A_i`` is accepted
    for signature parity only.

    Returns
    -------
    numpy.ndarray, shape (7,) -> [L, A_h, nan, k_2, nan, c_1, c_2]
    """
    v_h = global_params[8]

    k_2 = v_h * L / A_h * 60.0     # Sp -> channel rate  [1/min]
    c_1 = 0.001 / 60.0             # rainfall mm/hr -> m/min
    c_2 = A_h / 60.0                       # m/min * A_h -> m^3/s

    return np.array([L, A_h, np.nan, k_2, np.nan, c_1, c_2])


def precalculations_basic(L, A_h, A_i, global_params):
    """Per-link parameters for models that build their own rates (NZV1, Xinanjiang).

    Same output layout as :func:`precalculations`. These models build their
    own release coefficients from ``L``, ``A_h`` and the velocities in
    ``global_params``,
    so ``invtau``, ``k_2`` and ``k_i`` are NaN (unused). ``A_i`` is accepted
    for signature parity only.

    Returns
    -------
    numpy.ndarray, shape (7,) -> [L, A_h, nan, nan, nan, c_1, c_2]
    """
    c_1 = 0.001 / 60.0             # rainfall mm/hr -> m/min
    c_2 = A_h / 60.0               # m/min * A_h -> m^3/s

    return np.array([L, A_h, np.nan, np.nan, np.nan, c_1, c_2])


def precalculations_constant_runoff(L, A_h, A_i, global_params):
    """Per-link parameters for the ConstantRunoff model (asynch 190 hillslope).

    Same output layout as :func:`precalculations`. ``k_2`` is the runoff
    (fast) storage rate; the baseflow rate ``k_3`` is built in the flux
    function from ``v_g``. No channel routing, so ``invtau`` and ``k_i`` are
    NaN. ``A_i`` is accepted for signature parity only.

    Returns
    -------
    numpy.ndarray, shape (7,) -> [L, A_h, nan, k_2, nan, c_1, c_2]
    """
    v_h = global_params[1]

    k_2 = v_h * L / A_h * 60.0     # s_p -> channel rate  [1/min]
    c_1 = 0.001 / 60.0             # rainfall mm/hr -> m/min
    c_2 = A_h / 60.0               # m/min * A_h -> m^3/s

    return np.array([L, A_h, np.nan, k_2, np.nan, c_1, c_2])


class ArrayForcing:
    """Piecewise-constant forcing driven by per-record arrays.

    Mirrors Asynch's treatment of forcings: each record (e.g. one hour) holds
    its value constant over the whole interval ``dt``. Times before the first
    record use record 0; times at/after the last record hold the final value.

    Parameters
    ----------
    rain : 1-D array_like
        Rainfall per record [mm/hr].
    epot : float or 1-D array_like
        Potential evap [mm/month]. A scalar is broadcast to every record;
        an array must match ``len(rain)``.
    dt : float
        Record length in MINUTES (60.0 for hourly records).
    """

    def __init__(self, rain, epot, dt=60.0):
        self.rain = np.asarray(rain, dtype=float)
        self.n = self.rain.size
        if np.ndim(epot) == 0:
            self.epot = np.full(self.n, float(epot))
        else:
            self.epot = np.asarray(epot, dtype=float)
            if self.epot.size != self.n:
                raise ValueError("epot array must match rain length")
        self.dt = float(dt)

    def __call__(self, t):
        i = int(t // self.dt)                 # which record contains time t
        i = 0 if i < 0 else (self.n - 1 if i >= self.n else i)
        return self.rain[i], self.epot[i]

    @property
    def t_end(self):
        """End of the forcing series [min]."""
        return self.n * self.dt
