# Flux analysis for any model

The playground notebook analyses a model run in four ways:

1. **Flux time series**: every internal flux over the run.
2. **Water balance**: input − outputs − Δstorage ≈ 0.
3. **One-at-a-time (OAT) sensitivity**: peak, time to peak and volume vs.
   each parameter, with elasticities.
4. **Interactive hydrograph sweep**: plotly overlay of hydrographs for a few
   values of each parameter.

All four are built on one function, `flux_timeseries`, so they work for any
model registered with a `fluxes=` companion (see
[adding_a_model.md](adding_a_model.md)). This page shows what to change to
point each analysis at a new model. The examples use the `LR1` model from
that guide.

---

## 1. Flux time series

```python
from models import flux_timeseries

flux = flux_timeseries(model, sol.t, sol.y, params, gp, forcing)
# -> {"rain_in": array(N), "e_h": array(N), "q_hl": array(N), "q_out": array(N)}
```

- It re-evaluates `model.fluxes` at every time in `t`, using the solved
  states. The returned keys are whatever your flux function returns.
- `y` may be shaped `(dim, N)` (as in `sol.y`) or `(N, dim)`.
- Pass the **same** `params`, `gp` and `forcing` objects you integrated with.
- To get finer resolution than the solver's own steps, evaluate the dense
  solution on a grid:

  ```python
  t_fine = np.linspace(sol.t[0], sol.t[-1], 5000)
  flux = flux_timeseries(model, t_fine, sol.sol(t_fine), params, gp, forcing)
  ```

  This needs `dense_output=True` in `solve_ivp`.

### Plotting

The plot cells in the notebook select fluxes by key. For a new model,
change only the key tuples. Nothing else is model-specific:

```python
t_h = sol.t / 60.0
fig, (ax_q, ax_s, ax_f) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

ax_q.plot(t_h, flux["q_out"], lw=2.5, label="q_out (hillslope)")
ax_q.set_ylabel("discharge [m³/s]")

for name in ("s_h",):                                  # <- your storages
    ax_s.plot(t_h, sol.y[model.state_names.index(name)], label=name)
ax_s.set_ylabel("storage [m]")

for key in ("q_hl", "e_h"):                            # <- your flux keys
    ax_f.plot(t_h, flux[key], label=key)
ax_f.set_ylabel("flux [m/min]")
ax_f.set_yscale("symlog", linthresh=1e-9)              # fluxes span many decades

for ax in (ax_q, ax_s, ax_f):
    ax.legend(); ax.grid(alpha=0.3)
ax_f.set_xlabel("time [h]")
```

Streamflow is the model's outflow flux in m³/s, `flux["q_out"]` in every
current model. Any single hillslope flux converts the same way, for example
`flux["q_hl"] * params[6]` (`params[6]` is `c_2`).

---

## 2. Choosing what to balance

Before writing the balance, sort your model's terms into three groups:

| Group | What goes in it | LR1 | TopLayer | CRB3 |
|---|---|---|---|---|
| **Inputs** (fluxes into the hillslope) | rain, irrigation, ... | `rain_in` | `rain_in` | `rain_in` |
| **Outputs** (fluxes leaving the hillslope) | flows to channel, evaporation | `q_hl`, `e_h` | `q_pl`, `q_sl`, `e_p`, `e_t`, `e_s` | `q_pL`, `q_sL`, `q_gL`, `e_b`, `e_v` |
| **Storages** (hillslope states, in m) | | `s_h` | `s_p`, `s_t`, `s_s` | `s_s`, `s_p`, `s_g` |

**Leave out internal transfers.** Fluxes between two hillslope storages
(TopLayer's `q_pt`, `q_ts`, CRB3's `q_sp`, `q_ss`, `q_sg`) cancel out. Also
leave out states that are not hillslope water: cumulative-runoff
accumulators such as CRB3's `Q` or ConstantRunoff's `surf_runoff` and
`total_runoff`.

---

## 3. Water balance

The notebook has a balance function for each model (`hlm_water_balance`,
`crb_water_balance`). They differ only in the three lists above, so a new
model can use this generic version:

```python
def water_balance(model, sol, params, gp, forcing,
                  inputs, outputs, storages, n_sub=60):
    """Hillslope mass balance [mm] over the whole run.

    Integrates the fluxes with the midpoint rule on the dense solution
    (n_sub sub-steps per hour). Requires solve_ivp(..., dense_output=True).
    """
    t0, t1 = sol.t[0], sol.t[-1]
    edges = np.linspace(t0, t1, int(np.ceil((t1 - t0) / 60.0 * n_sub)) + 1)
    tm, h = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    f = flux_timeseries(model, tm, sol.sol(tm), params, gp, forcing)

    I = {k: np.sum(f[k] * h) * 1e3 for k in (*inputs, *outputs)}   # m -> mm
    idx = [model.state_names.index(s) for s in storages]
    dS = (sol.y[idx, -1] - sol.y[idx, 0]).sum() * 1e3

    out = dict(I, dStorage=dS)
    out["residual"] = sum(I[k] for k in inputs) - sum(I[k] for k in outputs) - dS
    return out

pd.Series(water_balance(model, sol, params, gp, forcing,
                        inputs=["rain_in"],
                        outputs=["q_hl", "e_h"],
                        storages=["s_h"]), name="mm").round(4).to_frame()
```

The same call reproduces the existing models' balances:

```python
water_balance(mtop, sol, p, gp, forcing, ["rain_in"],
              ["q_pl", "q_sl", "e_p", "e_t", "e_s"], ["s_p", "s_t", "s_s"])
water_balance(mcrb, sol, p, gp, forcing, ["rain_in"],
              ["q_pL", "q_sL", "q_gL", "e_b", "e_v"], ["s_s", "s_p", "s_g"])
```

**What to expect.** On a synthetic two-storm event (110 mm of rain), every
model in the registry closes to within about 1e-3 mm. A residual that is a noticeable
fraction of the rain means `dydt` and `fluxes` disagree, a term is in the
wrong group, or a unit is off (see
[adding_a_model.md § 3](adding_a_model.md#3-common-pitfalls)).

Extra checks are easy to add. CRB3, for instance, also verifies that its
cumulative-runoff state `Q` matches the integrated outflow
(`ΔQ − (q_pL + q_sL + q_gL)`).

---

## 4. One-at-a-time sensitivity

The OAT cells need three model-specific pieces:

```python
# 1. index -> name for your global-parameter layout
LR1_GP_NAMES = {i: name for i, (name, unit) in enumerate(model.param_names)}

# 2. index -> sweep values (use np.geomspace for parameters spanning decades)
lr1_sweep_ranges = {
    0: np.linspace(0.005, 0.1, 11),   # v_h
    1: np.linspace(0.02, 0.5, 11),    # S_max
}

# 3. the response to measure: streamflow in m³/s at the solver times
def lr1_q(sol, params, gp):
    return flux_timeseries(model, sol.t, sol.y, params, gp, forcing)["q_out"]
```

Then reuse the notebook's metric loop with these names:

```python
def lr1_run(gp):
    p = precalculations_basic(L, A_h, A_i, gp)
    s = solve_ivp(model.dydt, (0.0, forcing.t_end), y0, args=(p, gp, forcing),
                  method="LSODA", rtol=1e-6, atol=1e-9, max_step=30.0)
    return s, p

def lr1_metrics(gp):
    """(peak q [m³/s], time to peak [h], volume [m³])"""
    s, p = lr1_run(gp)
    if not s.success:
        return np.nan, np.nan, np.nan
    q = lr1_q(s, p, gp)
    return q.max(), s.t[q.argmax()] / 60.0, np.trapezoid(q, s.t * 60.0)
```

The loop that builds the elasticity table (`%Δoutput / %Δparam`) and the
plots below it is the same code as the `hlm_` and `crb_` sections. Copy it
and replace the prefix. If some parameter combinations are physically
invalid (TopLayer requires `S_L < h_b` and `Ssmin < Ssmax`), add a
`_valid(gp)` filter as in the TopLayer section.

## 5. Interactive hydrograph sweep

The plotly cell needs only:

- a dict `index -> (values, matplotlib colormap name)`, with about 5 values
  per parameter, and
- a `_run_q(gp) -> (t_hours, q)` function built from `lr1_run` and `lr1_q`.

Everything else in the cell (rain bars on a secondary axis, baseline in
black, legend groups per parameter) is model-independent.

---

## 6. Adding a model section to the notebook: checklist

1. Add a markdown header and a state and units description. Use a unique
   variable prefix (e.g. `lr1_`) so sections don't overwrite each other.
2. Get the model with `get_model`, copy its default `GLOBAL_PARAMS_*`, set
   `y0`, and `assert len(y0) == model.dim`.
3. Write a `<prefix>_run(gp)` wrapper around `solve_ivp` (or call
   `models.simulate`)
   (`dense_output=True`).
4. Compute `flux_timeseries`, and plot it with your own state and flux keys.
5. Run `water_balance` with your input, output and storage lists, and check
   the residual.
6. Run OAT sensitivity: names dict, sweep ranges, metric function.
7. Optionally, add the interactive hydrograph sweep.

To compare models directly, run them with the **same `forcing` object and
link geometry** (`L`, `A_h`, `A_i`). The CRB3 section does this with the
TopLayer section's forcing.
