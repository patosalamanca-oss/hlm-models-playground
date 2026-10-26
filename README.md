# HLM models playground

A small Python sandbox for prototyping and comparing **hillslope-link
hydrological models** (the Hillslope Link Model family run by
[ASYNCH](https://github.com/Iowa-Flood-Center/asynch)) on a single link,
before committing to an implementation in the C solver.

Every model is a Python right-hand side that `scipy.integrate.solve_ivp` can
integrate. Each model also exposes its **internal fluxes** (infiltration,
percolation, runoff, evaporation, ...), so after a run you can plot every
flux, check that the water balance closes, and run parameter-sensitivity
sweeps, all from the same equations the solver integrated.

## Contents

| File | What it is |
|---|---|
| [`models.py`](models.py) | Model registry and the right-hand sides of all models (`TopLayer`, `CRB3`, `NewZealandV1`, `ConstantRunoff`, `Xinanjiang`, `TankSMA`), plus `flux_timeseries` and `simulate`. |
| [`app.py`](app.py) | **Hillslope model explorer** web app: define the hillslope and a 24 h hyetograph, pick up to 3 models, edit their parameters and compare hydrographs and storages. |
| [`setup.py`](setup.py) | Model-independent helpers: `precalculations_*` (per-link parameters for each model), `ArrayForcing` (hourly rain/evap forcing) and default global-parameter vectors. |
| [`setup_Playground_HLM_CollieRB3.ipynb`](setup_Playground_HLM_CollieRB3.ipynb) | Single-link runs of TopLayer and CRB3: hydrograph, storages, fluxes, water balance, one-at-a-time sensitivity and interactive hydrograph sweeps. |
| [`docs/adding_a_model.md`](docs/adding_a_model.md) | **How to add a new model** to the registry. |
| [`docs/flux_analysis.md`](docs/flux_analysis.md) | **How to run the flux analysis** (flux plots, water balance, sensitivity) for any model, including a new one. |

> `setup.py` is a plain helper module, not a packaging script. Don't run
> `pip install .` on it.

## Quick start

```bash
git clone <this repo> && cd hlm-models-playground
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab setup_Playground_HLM_CollieRB3.ipynb
```

Minimal run from Python:

```python
import numpy as np
from scipy.integrate import solve_ivp
from models import get_model, flux_timeseries
from setup import precalculations_toplayer, ArrayForcing, GLOBAL_PARAMS_TOPLAYER

model = get_model("TopLayer")
gp = GLOBAL_PARAMS_TOPLAYER.copy()
params = precalculations_toplayer(L=350.0, A_h=133356.8, A_i=50.0, global_params=gp)

rain = np.zeros(240); rain[11:17] = [5, 10, 17, 10, 8, 3]    # mm/hr, hourly
forcing = ArrayForcing(rain, epot=50.0, dt=60.0)              # epot in mm/month, dt in minutes

y0 = [0.0, 0.05, 0.20]                                        # see model.state_names
sol = solve_ivp(model.dydt, (0.0, forcing.t_end), y0,
                args=(params, gp, forcing), method="LSODA",
                rtol=1e-6, atol=1e-9, max_step=30.0, dense_output=True)

fluxes = flux_timeseries(model, sol.t, sol.y, params, gp, forcing)
print(model.state_names, list(fluxes))
q = fluxes["q_out"]                                           # streamflow [m^3/s]
```

Or in one call, with the model's defaults: `models.simulate(model, L, A_h,
A_i, forcing, t_end)`.

## Hillslope model explorer (web app)

```bash
streamlit run app.py
```

This opens a browser page. Set the hillslope (`A_h`, `L_h`, upstream area
`A`) and the hourly rainfall in the sidebar, then tick up to three models.
Each ticked model expands to show its parameters and initial state,
pre-filled with defaults. The plots update on every change:

1. rainfall (inverted bars),
2. simulated streamflow per model,
3. storages per model (one colour per model, one line style per storage).

Hover, zoom and click legend entries to explore. A table below gives the
peak, time to peak and outflow volume.

**Optional channel routing.** Turn on *Route through the channel* in the
sidebar to route each model's hillslope outflow through the HLM non-linear
channel reservoir (`models.route_channel`):
`dq/dt = τ⁻¹ q^λ1 (Q_out − q)`, `τ⁻¹ = 60 v_0 A^λ2 / ((1 − λ1) L_d)`.
Here `A` is the upstream area and `L_d` the downstream channel length the
water travels. The routed hydrographs are drawn dashed, in each model's
colour, and the table adds the routed peak and the lag between the two
peaks. New models appear automatically once
their registration carries the metadata described in
[adding_a_model.md](docs/adding_a_model.md#making-the-model-show-up-in-the-web-app).

## Conventions

- **Time** is in **minutes**. Forcing records are piecewise constant over
  `dt` (60 min for hourly data), as in ASYNCH.
- **Storages** are in **m**. **Hillslope fluxes** are in **m/min**.
  Streamflow (the hillslope outflow `q_out`) is in **m³/s**.
- **No channel routing.** All models are hillslope runoff models; routing
  belongs to the HLM channel solver.
- **Rain** is given in mm/hr and **potential evaporation** in mm/month.
  `_read_forcing` converts both.
- **Per-link parameters** come from a `precalculations*` function and always
  have the layout `[L, A_h, invtau, k_2, k_i, c_1, c_2]`.

## Available models

| Name | States | Summary |
|---|---|---|
| `TopLayer` | `s_p, s_t, s_s` | Hillslope of ASYNCH models 254/255: ponded / top-layer / sub-surface storages, saturation-excess infiltration and a non-linear sub-surface release (`Ssmin`, `Ssmax`, `Ss_coef`, `Ss_exp`). Streamflow is the `q_out` flux. |
| `CRB3` | `Q, s_s, s_p, s_g` | Collie River Basin 3 (MARRMoT-style): soil, ponded and groundwater storages. No channel routing; streamflow is the `q_out` flux. |
| `NewZealandV1` | `Q, s_m, s_p` | Soil-matrix (Sm) and ponded (Sp) storages; Sp fills by smoothed saturation excess. Sm drains by sub-surface (above field capacity) and base flow. No channel routing; streamflow is the `q_out` flux. |
| `ConstantRunoff` | `s_p, s_a, surf_runoff, total_runoff` | Asynch 190 hillslope (port of UMN_HLM_Coupled `model_ConstantRunoff.hpp`): rain split by a constant coefficient `rc` between a fast and a slow linear reservoir. Streamflow is the `q_out` flux. |
| `Xinanjiang` | `s_w, s_f, s_i, s_g, s_p` | MARRMoT model 28 (port of FindBestModel `step_model_28`) with an added ponded storage `s_p`: surface runoff `RS + RB` drains as `K_p·Sp^a_1` instead of reaching the channel instantly. Rates `kI, kG, cI, cG` in 1/d. Streamflow is the `q_out` flux. |
| `TankSMA` | `s_1, x_s, s_2, s_3, s_4` | Tank Model - SMA, MARRMoT model 38 (port of FindBestModel `step_model_38`): four stacked tanks with side outlets plus secondary soil moisture `x_s` and capillary rise. Threshold outlets use the literal `A·max(S − t, 0)`, and the evaporation cascade uses a ramp that is 0 for an empty tank. Rates in 1/d (`k1`, `k2` in mm/d). Streamflow is the `q_out` flux. |

To list them at runtime: `from models import MODELS; print(MODELS.keys())`.
