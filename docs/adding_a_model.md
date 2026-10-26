# Adding a new model

This guide walks through adding a model to [`models.py`](../models.py) so
that it can be run in the playground and analysed with the flux tools
described in [flux_analysis.md](flux_analysis.md).

The running example is **`LR1`**, a deliberately simple model: one linear
hillslope reservoir whose outflow is the hillslope streamflow `q_out`
[m³/s]. Like every model in the registry, it is a runoff model: channel
routing is not part of it.

---

## 1. How a model is structured

Every model is made of **two functions** plus a **registration**:

```
fluxes(t, y, params, global_params, forcing)  ->  dict[str, float]
        │   every internal flux and rate, computed once
        ▼
dydt(t, y, params, global_params, forcing, parents_q=0, parents_qb=0)  ->  np.ndarray
        │   assembles dy/dt from the flux dict (this is what solve_ivp integrates)
        ▼
@register_model(name, dim, state_names, description, fluxes=...)
            stores both in MODELS[name] as an HLMModel
```

The key rule is that **`dydt` must compute its derivatives from the dict
returned by `fluxes`**, and must never re-derive a flux on its own. The
post-run analysis (`flux_timeseries`) calls the same `fluxes` function, so
the fluxes you plot and integrate are exactly the ones the solver used.

### The call contract

| Argument | Meaning |
|---|---|
| `t` | Time [min]. Used only to sample the forcing. |
| `y` | State vector, length `dim`, in the order of `state_names`. |
| `params` | Per-link parameters `[L, A_h, invtau, k_2, k_i, c_1, c_2]` from a `precalculations*` function in [`setup.py`](../setup.py). |
| `global_params` | 1-D NumPy array of global parameters. **You** choose the index layout for your model. |
| `forcing` | `ArrayForcing` (callable `forcing(t) -> (rain_mm_hr, epot_mm_month)`) or a constant 2-tuple. Read it with `_read_forcing(forcing, t)`, which returns rain in mm/hr and `e_pot` already in m/min. |
| `parents_q`, `parents_qb` | Summed upstream channel discharge and base flow [m³/s], kept for signature parity with the HLM solver. They are 0 for a single link, and the current models ignore them. Accept them anyway, so the signature stays uniform. |

### Units

- Storages are in m. Hillslope fluxes are in m/min (rain: `rain_mm_hr * c_1`).
- To convert a hillslope flux to channel inflow in m³/s, multiply by `c_2`
  (`= A_h / 60`).
- Release rates such as `k_2 = v_h · L / A_h · 60` are in 1/min.

---

## 2. Step by step

### Step 1: define the global-parameter layout

Decide what each index of `global_params` means, then add a **documented
default vector** to [`setup.py`](../setup.py), next to the existing ones.
Also document the layout in that module's docstring.

```python
# setup.py
GLOBAL_PARAMS_LR1 = np.array([
    0.02,        # 0  v_h        hillslope velocity [m/s]
    0.10,        # 1  S_max      storage at which ET is unrestricted [m]
])
```

> **Per-link parameters:** every model has a `precalculations_<name>()` in
> `setup.py` that returns the 7-element layout
> `[L, A_h, invtau, k_2, k_i, c_1, c_2]`, with `np.nan` in the slots it
> doesn't use. `precalculations_basic` fills only `L, A_h, c_1, c_2`. It
> suits models like LR1, NewZealandV1 and Xinanjiang that build their
> release rates from `L`, `A_h` and the global parameters inside the flux
> function. To precompute rates once per link instead, write your own, as
> `precalculations_toplayer` does for `k_2` and `k_i`.

### Step 2: write the flux function

Put it in `models.py`, in a new section following the existing ones. Return
**every** quantity you might want to plot or balance, including rates such
as `k_t`, and not only the terms that appear in `dydt`.

```python
# ----------------------------------------------------------------------------
# LR1 -- single linear hillslope reservoir
# ----------------------------------------------------------------------------
def fluxes_lr1(t, y, params, global_params, forcing):
    v_h   = global_params[0]                      # [m/s]
    S_max = global_params[1]                      # [m]

    rain_mm_hr, e_pot = _read_forcing(forcing, t)
    L, A_h, invtau, k_2, k_i, c_1, c_2 = params
    s_h = max(y[0], 0.0)                          # guard against tiny negatives
    k_h = v_h * L / A_h * 60.0                    # release rate        [1/min]

    q_hl = k_h * s_h
    return {
        "rain_in": rain_mm_hr * c_1,              # rain -> hillslope   [m/min]
        "e_h":     e_pot * min(s_h / S_max, 1.0), # evaporation         [m/min]
        "q_hl":    q_hl,                          # hillslope -> channel[m/min]
        "q_out":   q_hl * c_2,                    # streamflow          [m^3/s]
    }
```

Guidelines:

- **Use the same key names as the existing models wherever the meaning is
  the same.** `rain_in` is the conventional rainfall input. The notebook
  analysis code looks up fluxes by key.
- Name transfer fluxes `q_<from><to>`, e.g. `q_pt` = ponded → top,
  `q_sl` = sub-surface → link (channel).
- Clip states with `max(y[i], 0.0)` before using them in powers or
  divisions. Adaptive solvers can overshoot slightly below zero.
- Keep the function free of side effects. It is called thousands of times.

### Step 3: write the RHS and register it

```python
@register_model(
    "LR1", dim=1,
    state_names=("s_h",),
    description="Single linear hillslope reservoir. Streamflow is the "
                "'q_out' flux [m^3/s].",
    fluxes=fluxes_lr1,
)
def model_lr1(t, y, params, global_params, forcing,
              parents_q=0.0, parents_qb=0.0):
    # parents_q / parents_qb are ignored: routing is not part of this model.
    f = fluxes_lr1(t, y, params, global_params, forcing)
    ans = np.zeros(1)
    ans[0] = f["rain_in"] - f["q_hl"] - f["e_h"]     # s_h
    return ans
```

Checklist for the registration:

- `name` must be unique. It is the key you pass to `get_model`.
- `dim` must equal `len(state_names)` and the length of the returned array.
- The order of `state_names` must match the indices used in `dydt`. The flux
  analysis uses `model.state_names.index(...)` to find storages.
- Always pass `fluxes=`. Without it, `flux_timeseries` raises a
  `ValueError`.

#### Making the model show up in the web app

The explorer app ([`app.py`](../app.py)) lists every model whose
registration also carries simulation metadata. Add these keywords to
`register_model`:

```python
@register_model(
    "LR1", dim=1, state_names=("s_h",), description="...",
    fluxes=fluxes_lr1,
    precalc=precalculations_basic,            # (L, A_h, A_i, gp) -> params
    param_names=(("v_h", "m/s"), ("S_max", "m")),
    default_params=GLOBAL_PARAMS_LR1,         # same length as param_names
    y0=(0.01,),                               # default initial state, length dim
    storages=("s_h",),                        # states plotted as storages [m]
    streamflow="q_out",                       # flux key or state name [m^3/s]
)
```

`streamflow` is the flux key of the hillslope outflow (`"q_out"` in every
current model). It can also name a state, for a model that carries its own
discharge state. The same metadata lets you run the model in one call with
`models.simulate(...)`.

To enable the app's **Equations** button, also pass the equations as data:

```python
    balance={                                  # storage: (inflows, outflows)
        "s_h": (("rain_in",), ("q_hl", "e_h")),
    },
    flux_eqs={                                 # flux key: (symbol, rhs), LaTeX
        "rain_in": (r"P",      r"c_1\,p(t)"),
        "e_h":     (r"E_h",    r"E_{pot}\,\min(S_h / S_{max},\,1)"),
        "q_hl":    (r"q_{hl}", r"k_h\,S_h"),
        "q_out":   (r"Q_{out}", r"c_2\,q_{hl}"),
    },
    extra_eqs=(                                # definitions, LaTeX
        (r"k_h", r"60\,v_h\,\frac{L}{A_h}"),
    ),
```

The app builds the balance equations (dS/dt = inflows − outflows) from
`balance`. It also draws a **flow diagram** from it (`models.flow_edges`):
a flux leaving one storage and entering another becomes an arrow between
them. A flux that only enters storages comes from *Rain*. One that only
leaves goes to *ET* if its key starts with `e`, otherwise to *Channel*. When
that guess is wrong, for example a flux split between two targets, override
single fluxes with `diagram=`, as CRB3 does:

```python
    diagram={"q_ss": None,                    # hide the un-split flux
             "q_sg": ("s_s", "s_g"),          # (from, to); Rain/ET/Channel
             "q_sL": ("s_s", CHANNEL)},       # are models.RAIN/ET/CHANNEL
```

Every key in `balance` must be a key of `flux_eqs` and of the
dict your `fluxes` function returns, and the balance must match `dydt`
exactly. Check it numerically: at a few random states, the sum of inflows
minus outflows should equal the corresponding `dydt` entry.

> **Variants of an existing model:** if your model only changes one or two
> fluxes of an existing one (say TopLayer), copy its flux function and
> registration, change those lines, and give it a new name. Keep the flux
> keys and state names. Its `balance`, `flux_eqs` and the notebook's
> analysis cells for that model then carry over with little change.

### Step 4: smoke-test it

Run this in the notebook. Because `%autoreload 2` is on, edits to
`models.py` and `setup.py` are picked up without restarting the kernel.

```python
import numpy as np
from scipy.integrate import solve_ivp
from models import get_model, flux_timeseries
from setup import precalculations_basic, ArrayForcing, GLOBAL_PARAMS_LR1

m  = get_model("LR1")
gp = GLOBAL_PARAMS_LR1.copy()
p  = precalculations_basic(350.0, 133356.8, 50.0, gp)

rain = np.zeros(240); rain[11:17] = [5, 10, 17, 10, 8, 3]
forcing = ArrayForcing(rain, 50.0, dt=60.0)

y0 = np.array([0.01])
assert len(y0) == m.dim
assert len(m.dydt(0.0, y0, p, gp, forcing)) == m.dim

sol = solve_ivp(m.dydt, (0.0, forcing.t_end), y0, args=(p, gp, forcing),
                method="LSODA", rtol=1e-6, atol=1e-9, max_step=30.0,
                dense_output=True)
assert sol.success, sol.message
f = flux_timeseries(m, sol.t, sol.y, p, gp, forcing)
print({k: v.max() for k, v in f.items()})
```

Then close the water balance as described in
[flux_analysis.md § 3](flux_analysis.md#3-water-balance). A residual close
to 0 mm is the best evidence that `fluxes` and `dydt` are consistent. With
the forcing above, `LR1` closes to about 1e-4 mm out of 53 mm of rain.

### Step 5: document it

- Add a row to the **Available models** table in the [README](../README.md).
- Describe the global-parameter layout in the `setup.py` docstring.
- If the model is non-trivial, add a notebook section for it, following
  [flux_analysis.md](flux_analysis.md).

---

## 3. Common pitfalls

| Symptom | Likely cause |
|---|---|
| `solve_ivp` is extremely slow or reports "step size too small" | Stiff parameters, e.g. a very fast drainage coefficient, or an exponent < 1 on a storage that reaches 0 (infinite derivative). Try `method="LSODA"` or `"Radau"`, or reconsider the defaults. The CRB3 section of the notebook overrides `A` and `B2` for exactly this reason. |
| Water-balance residual is large | `dydt` uses a flux not returned by `fluxes`, a flux has the wrong units (mm vs m, hr vs min), or a storage is missing from the balance list. |
| `ValueError: ... has no flux function registered` | `fluxes=` was not passed to `register_model`. |
| `KeyError` in the notebook plotting cells | Those cells use another model's flux names (TopLayer's `q_pl`, `q_sl`, ...). Change the key tuples to your model's names. See [flux_analysis.md](flux_analysis.md). |
| `NaN` in output | A negative storage raised to a fractional power. Clip it with `max(y[i], 0.0)`. |
