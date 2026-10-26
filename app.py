"""Hillslope model explorer -- compare hydrograph responses of up to 3 models.

Run with::

    streamlit run app.py

Every model in ``models.MODELS`` whose registration carries simulation
metadata (see ``HLMModel.runnable``) shows up here automatically.
"""

import html
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from models import MODELS, simulate, route_channel, flow_edges, RAIN, ET, CHANNEL
from setup import ArrayForcing

MAX_MODELS = 3
MAX_RAIN_HOURS = 24
MODEL_COLORS = ("#1f77b4", "#d62728", "#2ca02c")
STORAGE_DASHES = ("solid", "dot", "dashdot", "longdashdot", "longdash",
                  "2px,4px,8px,4px")    # "dash" is reserved for routed Q
DEFAULT_RAIN = [0, 0, 5, 10, 17, 10, 8, 3] + [0] * (MAX_RAIN_HOURS - 8)

st.set_page_config(page_title="Hillslope model explorer", layout="wide")


def _pin_results_panel():
    """Keep the results column in view while the page scrolls (wide screens
    only; on narrow screens the columns stack and normal scrolling is better)."""
    st.markdown("""
    <style>
    @media (min-width: 640px) {
      div[data-testid="stColumn"]:has(.st-key-results_panel) {
        position: sticky;
        top: 4rem;
        align-self: flex-start;
      }
    }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _step(v):
    """A sensible +/- step for a number_input, based on the value's magnitude."""
    v = abs(v)
    return 10.0 ** (math.floor(math.log10(v)) - 1) if v > 0 else 0.01


def _reset_model(name):
    for k in [k for k in st.session_state if k.startswith(f"{name}::")]:
        del st.session_state[k]


@st.cache_data(show_spinner=False)
def run_model(name, L, A_h, A_i, rain, epot, sim_hours, gp, y0):
    model = MODELS[name]
    rain = np.r_[rain, np.zeros(max(sim_hours - len(rain), 0))]
    forcing = ArrayForcing(rain, epot, dt=60.0)
    t_end = sim_hours * 60.0
    return simulate(model, L, A_h, A_i, forcing, t_end, global_params=gp,
                    y0=y0, t_eval=np.arange(0.0, t_end + 1e-9, 5.0))


@st.cache_data(show_spinner=False)
def run_routing(t, q, routing):
    return route_channel(t, q, **routing)


def model_inputs(model):
    """Render the parameter / initial-state inputs; return (gp, y0) tuples."""
    gp = []
    cols = st.columns(2)
    for i, ((pname, unit), default) in enumerate(
            zip(model.param_names, model.default_params)):
        with cols[i % 2]:
            gp.append(st.number_input(
                f"{pname} [{unit}]", value=float(default), step=_step(default),
                format="%g", key=f"{model.name}::gp{i}"))

    st.caption("Initial state")
    y0 = []
    cols = st.columns(2)
    for i, (sname, default) in enumerate(zip(model.state_names, model.y0)):
        with cols[i % 2]:
            y0.append(st.number_input(
                sname, value=float(default), step=_step(default),
                format="%g", key=f"{model.name}::y0{i}"))

    st.button("Reset to defaults", key=f"{model.name}::reset",
              on_click=_reset_model, args=(model.name,))
    return tuple(gp), tuple(y0)


def _toggle_equations(name):
    key = f"eqs::{name}"
    st.session_state[key] = not st.session_state.get(key, False)


def _tex_storage(name):
    """'s_m' -> 'S_{m}'."""
    head, _, sub = name.partition("_")
    return f"{head.upper()}_{{{sub}}}" if sub else head.upper()


def _aligned(lines):
    return r"\begin{aligned}" + r" \\[6pt] ".join(lines) + r"\end{aligned}"


def _html_symbol(tex):
    """'q_{mp}' -> 'q' + small 'mp' for Graphviz HTML labels (<SUB> renders
    misaligned in Streamlit's Graphviz, so the subscript is just smaller)."""
    m = re.fullmatch(r"([A-Za-z]+)_\{?([A-Za-z0-9,]+)\}?", tex)
    return (f'{m[1]}<FONT POINT-SIZE="8">{m[2]}</FONT>' if m
            else html.escape(tex))


def _mm(v):
    return f"{v:.3g} mm" if abs(v) >= 1e-3 else "≈0 mm"


def flow_diagram(model, res=None):
    """DOT source: storages as boxes, fluxes as arrows. With a simulation
    result ``res``, arrow widths scale with each flux's total over the run."""
    edges = flow_edges(model)
    totals = None
    if res is not None:
        t, f = res["t"], res["fluxes"]
        totals = {k: float(np.trapezoid(f[k], t)) * 1e3 for k, _, _ in edges}  # mm
        biggest = max(max(totals.values()), 1e-12)

    lines = ['digraph {', 'rankdir=TB; bgcolor="transparent"; nodesep=0.5; ranksep=0.6;',
             'node [fontname="Helvetica", fontsize=12, fontcolor="#1f2937"];',
             'edge [fontname="Helvetica", fontsize=10, fontcolor="#374151"];']
    used = {n for _, a, b in edges for n in (a, b)}
    for node, fill in ((RAIN, "#dbeafe"), (ET, "#ffedd5"), (CHANNEL, "#cffafe")):
        if node in used:
            lines.append(f'"{node}" [shape=ellipse, style=filled, '
                         f'fillcolor="{fill}", color="#9ca3af"];')
    for s in model.balance:
        label = _html_symbol(_tex_storage(s))
        if res is not None:
            peak = res["y"][model.state_names.index(s)].max() * 1e3
            label += f'<BR/><FONT POINT-SIZE="9">peak {_mm(peak)}</FONT>'
        lines.append(f'"{s}" [shape=box, style="rounded,filled", '
                     f'fillcolor="#f3f4f6", color="#6b7280", label=<{label}>];')
    lines.append(f'{{rank=source; "{RAIN}"; "{ET}";}}' if ET in used
                 else f'{{rank=source; "{RAIN}";}}')
    if CHANNEL in used:
        lines.append(f'{{rank=sink; "{CHANNEL}";}}')

    for k, a, b in edges:
        label = _html_symbol(model.flux_eqs[k][0]) if k in model.flux_eqs else k
        width = 1.5
        if totals is not None:
            label += f"<BR/>{_mm(totals[k])}"
            width = 1.0 + 7.0 * totals[k] / biggest
        style = ', style=dashed, color="#c2410c"' if b == ET else ', color="#1d4ed8"'
        lines.append(f'"{a}" -> "{b}" [label=<{label}>, penwidth={width:.2f}{style}];')
    lines.append("}")
    return "\n".join(lines)


def equations_panel(model, res=None):
    """Flow diagram plus mass-balance, flux and auxiliary equations of one model."""
    sym = lambda k: model.flux_eqs[k][0]
    with st.container(border=True):
        st.markdown(f"#### {model.name} equations")
        st.caption("Flow diagram: " + (
            "arrow width ∝ total flux over the run [mm]; dashed = evaporation."
            if res is not None else
            "select the model to scale the arrows by its simulated fluxes."))
        st.graphviz_chart(flow_diagram(model, res))
        st.caption("Mass balance [m/min]")
        lines = []
        for storage, (ins, outs) in model.balance.items():
            rhs = " + ".join(sym(k) for k in ins) + "".join(f" - {sym(k)}" for k in outs)
            lines.append(rf"\frac{{d{_tex_storage(storage)}}}{{dt}} &= {rhs}")
        st.latex(_aligned(lines))
        st.caption("Fluxes and rates")
        st.latex(_aligned(f"{lhs} &= {rhs}" for lhs, rhs in model.flux_eqs.values()))
        if model.extra_eqs:
            st.caption("Definitions")
            st.latex(_aligned(f"{lhs} &= {rhs}" for lhs, rhs in model.extra_eqs))
        st.caption("p(t): rainfall [mm/h]; E_pot: potential ET [m/min]; "
                   "storages in m, hillslope fluxes in m/min, Q_out in m³/s.")


def make_figure(rain, results, routed, sim_hours, log_q, log_storage):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.2, 0.4, 0.4], vertical_spacing=0.04)

    hours = np.arange(len(rain))
    fig.add_trace(go.Bar(x=hours + 0.5, y=rain, width=1.0, name="Rainfall",
                         marker_color="#4a90c2",
                         hovertemplate="hour %{x:.0f}: %{y} mm/hr<extra></extra>"),
                  row=1, col=1)

    for (name, model, res), color in zip(results, MODEL_COLORS):
        t_h = res["t"] / 60.0
        fig.add_trace(go.Scatter(
            x=t_h, y=res["q"], name=f"{name} · Q", legendgroup=name,
            legendgrouptitle_text=name, line=dict(color=color, width=2.5),
            hovertemplate="%{y:.4g} m³/s<extra>" + name + "</extra>"),
            row=2, col=1)
        if name in routed:
            fig.add_trace(go.Scatter(
                x=t_h, y=routed[name], name=f"{name} · Q routed", legendgroup=name,
                line=dict(color=color, width=2.5, dash="dash"),
                hovertemplate="%{y:.4g} m³/s<extra>" + name + " routed</extra>"),
                row=2, col=1)
        for sname, dash in zip(model.storages, STORAGE_DASHES):
            s_mm = res["y"][model.state_names.index(sname)] * 1e3
            fig.add_trace(go.Scatter(
                x=t_h, y=s_mm, name=f"{name} · {sname}", legendgroup=name,
                line=dict(color=color, dash=dash, width=1.8),
                hovertemplate="%{y:.4g} mm<extra>" + f"{name} {sname}</extra>"),
                row=3, col=1)

    fig.update_yaxes(title_text="Rain [mm/hr]", autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text="Streamflow [m³/s]", row=2, col=1,
                     type="log" if log_q else "linear",
                     rangemode="normal" if log_q else "tozero")
    fig.update_yaxes(title_text="Storage [mm]", row=3, col=1,
                     type="log" if log_storage else "linear")
    fig.update_xaxes(range=[0, sim_hours])
    fig.update_xaxes(title_text="Time [h]", row=3, col=1)
    fig.update_layout(height=570, hovermode="x unified", bargap=0,
                      margin=dict(l=60, r=20, t=10, b=40),
                      legend=dict(groupclick="toggleitem"))
    return fig


def summary_table(results, routed, rain, A_h):
    rain_vol = float(np.sum(rain)) * 1e-3 * A_h          # mm/hr * 1 h -> m^3
    rows = []
    for name, _, res in results:
        q, t = res["q"], res["t"]
        k = int(np.argmax(q))
        vol = float(np.trapezoid(q, t * 60.0))           # m^3/s * s
        rows.append({
            "Model": name,
            "Peak Q [m³/s]": q[k],
            "Peak time [h]": t[k] / 60.0,
            "Volume [m³]": vol,
            "Out / rain [-]": vol / rain_vol if rain_vol > 0 else np.nan,
        })
        if name in routed:
            qr = routed[name]
            kr = int(np.argmax(qr))
            rows[-1]["Routed peak [m³/s]"] = qr[kr]
            rows[-1]["Lag [h]"] = (t[kr] - t[k]) / 60.0
    return pd.DataFrame(rows).set_index("Model")


# ----------------------------------------------------------------------------
# Sidebar: hillslope, simulation and rainfall
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Hillslope")
    A_h = st.number_input("Hillslope area A_h [m²]", min_value=1.0,
                          value=133356.8, step=1000.0, format="%g")
    L = st.number_input("Channel length L_h [m]", min_value=1.0,
                        value=350.0, step=10.0, format="%g")
    A_i = st.number_input("Upstream area A [km²]", min_value=1e-3,
                          value=50.0, step=1.0, format="%g",
                          help="Drainage area at the outlet; used by the "
                               "channel routing (invtau).")

    st.header("Routing")
    route_on = st.toggle("Route through the channel", value=False,
                         help="Adds each model's routed hydrograph (dashed).")
    routing = None
    if route_on:
        c1, c2 = st.columns(2)
        v_0 = c1.number_input("v_0 [m/s]", min_value=1e-3, value=0.33,
                              step=0.01, format="%g")
        L_d = c2.number_input("L_d [m]", min_value=1.0, value=5000.0,
                              step=500.0, format="%g",
                              help="Downstream channel length the water "
                                   "travels before the outlet.")
        lambda_1 = c1.number_input("λ1 [-]", min_value=0.0, max_value=0.99,
                                   value=0.20, step=0.01, format="%g")
        lambda_2 = c2.number_input("λ2 [-]", min_value=-1.0, max_value=1.0,
                                   value=-0.10, step=0.01, format="%g")
        routing = dict(v_0=v_0, lambda_1=lambda_1, lambda_2=lambda_2,
                       L=L_d, A_i=A_i)
        st.latex(r"\frac{dq}{dt} = \tau^{-1} q^{\lambda_1}\,(Q_{out} - q)")
        st.latex(r"\tau^{-1} = \frac{60\,v_0\,A^{\lambda_2}}{(1-\lambda_1)\,L_d}")

    st.header("Simulation")
    sim_hours = st.number_input("Length [h]", min_value=MAX_RAIN_HOURS,
                                max_value=24 * 30, value=72, step=12)
    epot = st.number_input("Potential ET [mm/month]", min_value=0.0,
                           value=50.0, step=5.0)

    st.header("Rainfall")
    rain_df = st.data_editor(
        pd.DataFrame({"Hour": np.arange(1, MAX_RAIN_HOURS + 1),
                      "Rain [mm/hr]": np.array(DEFAULT_RAIN, dtype=float)}),
        hide_index=True, disabled=["Hour"], height=36 * (MAX_RAIN_HOURS + 1),
        column_config={"Rain [mm/hr]": st.column_config.NumberColumn(
            min_value=0.0, step=0.1, format="%.1f")},
        key="rain_table")
    rain = np.clip(rain_df["Rain [mm/hr]"].fillna(0.0).to_numpy(float), 0.0, None)
    st.caption(f"Total: {rain.sum():.1f} mm")


# ----------------------------------------------------------------------------
# Main: model selection | plots
# ----------------------------------------------------------------------------
st.title("Hillslope model explorer")

runnable = [m for m in MODELS.values() if m.runnable]
n_selected = sum(bool(st.session_state.get(f"use::{m.name}")) for m in runnable)

left, right = st.columns([1, 2], gap="large")

selected = []
with left:
    st.subheader(f"Models ({n_selected}/{MAX_MODELS})")
    for model in runnable:
        key = f"use::{model.name}"
        c_box, c_eq = st.columns([3, 2], vertical_alignment="center")
        on = c_box.checkbox(model.name, key=key, help=model.description,
                            disabled=n_selected >= MAX_MODELS
                                     and not st.session_state.get(key))
        if model.balance and model.flux_eqs:
            shown = st.session_state.get(f"eqs::{model.name}", False)
            c_eq.button("Equations", key=f"eqs_btn::{model.name}",
                        icon=":material/functions:",
                        type="primary" if shown else "secondary",
                        on_click=_toggle_equations, args=(model.name,))
        if on:
            with st.expander(f"{model.name} parameters", expanded=True,
                             key=f"{model.name}::expander"):
                st.caption(model.description)
                gp, y0 = model_inputs(model)
            selected.append((model, gp, y0))

eq_models = [m for m in runnable if st.session_state.get(f"eqs::{m.name}")]
if not eq_models:
    # Pinning a panel taller than the window would hide the equations.
    _pin_results_panel()

with right, st.container(key="results_panel"):
    results = []
    if not selected:
        st.info("Select up to three models on the left to simulate them.")
    else:
        for model, gp, y0 in selected:
            try:
                res = run_model(model.name, L, A_h, A_i, tuple(rain), epot,
                                int(sim_hours), gp, y0)
                results.append((model.name, model, res))
            except Exception as exc:                     # solver blow-ups etc.
                st.error(f"{model.name} failed: {exc}")

        routed = {}
        if routing is not None:
            for name, _, res in results:
                try:
                    routed[name] = run_routing(res["t"], res["q"], routing)
                except Exception as exc:
                    st.error(f"Routing {name} failed: {exc}")

        c1, c2 = st.columns(2)
        log_q = c1.toggle("Log scale for streamflow", value=False)
        log_storage = c2.toggle("Log scale for storages", value=False)
        st.plotly_chart(make_figure(rain, results, routed, int(sim_hours), log_q,
                                    log_storage),
                        width="stretch")
        if results:
            st.dataframe(summary_table(results, routed, rain, A_h)
                         .style.format("{:.4g}"))
            st.caption("Out / rain can exceed 1: outflow also includes "
                       "drainage of the initial storages. Lag = time "
                       "between the hillslope and the routed peak.")

    results_by_name = {name: res for name, _, res in results}
    for model in eq_models:
        equations_panel(model, results_by_name.get(model.name))
