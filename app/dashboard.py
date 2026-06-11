"""pitwall strategy dashboard (Streamlit).

Run:  PYTHONPATH=src streamlit run app/dashboard.py

A strategist-facing what-if tool: pick a race + grid slot, get a robust strategy
recommendation with a strategist-readable reason, sweep the whole pit-lap ×
compound space in milliseconds via the learned surrogate, inspect the risk/reward
frontier, watch the live Bayesian tyre belief on a real stint, and see how the
recommendation flips the instant a Safety Car deploys.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitwall.models import RaceModel  # noqa: E402
from pitwall.models.circuits import CIRCUITS  # noqa: E402
from pitwall.optimize import optimize  # noqa: E402
from pitwall.sim import ScenarioSet, build_field  # noqa: E402
from pitwall.types import Compound, Stint, Strategy  # noqa: E402

st.set_page_config(page_title="pitwall — F1 strategy engine", layout="wide")


@st.cache_data(show_spinner=False)
def get_rivals(circuit: str):
    return build_field(circuit, n_cars=20, seed=1)


@st.cache_data(show_spinner=False)
def get_scenarios(circuit: str, n: int):
    m = RaceModel.for_circuit(circuit)
    return ScenarioSet.sample(m.safety_car, m.config.n_laps, n, seed=7)


@st.cache_resource(show_spinner=False)
def get_surrogate():
    try:
        from pitwall.surrogate import load
        return load()
    except Exception:
        return None


def heatmap_surrogate(net, model, grid, delta, spread=1.5):
    """Sweep 1-stop pit-lap × compound-pair, predict mean position instantly."""
    from pitwall.surrogate import encode
    n_laps = model.config.n_laps
    pairs = [(Compound.SOFT, Compound.HARD), (Compound.SOFT, Compound.MEDIUM),
             (Compound.MEDIUM, Compound.HARD), (Compound.MEDIUM, Compound.SOFT),
             (Compound.HARD, Compound.MEDIUM), (Compound.HARD, Compound.SOFT)]
    laps = list(range(6, n_laps - 4))
    feats, meta = [], []
    for ci, (a, b) in enumerate(pairs):
        for p in laps:
            strat = Strategy([Stint(a, p), Stint(b, n_laps - p)])
            feats.append(encode(model, strat, focal_grid=grid, focal_delta=delta, pace_spread=spread))
            meta.append((ci, p))
    t0 = time.time()
    pred = net.predict(np.asarray(feats))
    dt = (time.time() - t0) * 1000
    Z = np.full((len(pairs), len(laps)), np.nan)
    for (ci, p), mp in zip(meta, pred["mean_pos"]):
        Z[ci, laps.index(p)] = mp
    labels = [f"{a.value[0]}-{b.value[0]}" for a, b in pairs]
    return laps, labels, Z, dt, len(feats)


# ----------------------------------------------------------------------------
st.title("🏁 pitwall — race-strategy engine")
st.caption("Physics calibrated on real F1 data · Monte-Carlo robust optimisation · "
           "learned surrogate for real-time what-ifs · live Bayesian tyre updating")

with st.sidebar:
    st.header("Race setup")
    circuit = st.selectbox("Circuit", list(CIRCUITS.keys()),
                           index=list(CIRCUITS.keys()).index("bahrain"),
                           format_func=lambda c: CIRCUITS[c].name)
    grid = st.slider("Starting grid position", 1, 20, 3)
    delta = st.slider("Car pace vs reference (s/lap, +slower)", -0.5, 2.5, 0.3, 0.05)
    objective = st.selectbox("Objective", ["podium", "expected", "win", "points", "robust"])
    scenarios = st.select_slider("Monte-Carlo scenarios", [200, 400, 800, 1500], value=400)

model = RaceModel.for_circuit(circuit)
prof = CIRCUITS[circuit]
rivals = get_rivals(circuit)
scen = get_scenarios(circuit, scenarios)
net = get_surrogate()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Laps", prof.n_laps)
c2.metric("Pit loss", f"{prof.pit_loss_s:.0f}s")
c3.metric("Overtaking", f"{prof.overtake_threshold_s:.2f} s/lap")
c4.metric("P(safety car)", f"{prof.p_safety_car:.0%}")

with st.spinner("Optimising strategy over scenarios…"):
    res = optimize(model, rivals, scen, circuit_id=circuit, focal_grid=grid,
                   focal_delta=delta, objective=objective, shortlist=12)
best = res.best

st.subheader("Recommendation")
rc1, rc2 = st.columns([1, 2])
with rc1:
    st.markdown(f"## `{best.strategy.label()}`")
    e = best.ensemble
    st.markdown(
        f"- **{best.strategy.n_stops}-stop**, pit lap(s) **{', '.join(map(str, best.strategy.pit_laps))}**\n"
        f"- Expected finish: **P{e.mean_position:.1f}**\n"
        f"- P(win) **{e.p_win:.0%}** · P(podium) **{e.p_podium:.0%}** · P(points) **{e.p_points:.0%}**\n"
        f"- Tail risk (CVaR10): **P{e.cvar_position():.1f}** · DNF **{e.dnf_rate:.0%}**"
    )
with rc2:
    runner = res.ranked[1]
    gap = runner.ensemble.mean_position - e.mean_position
    st.markdown("**Why this call**")
    st.markdown(
        f"- Beats the next-best `{runner.strategy.label()}` by **{gap:.2f}** positions on expected finish.\n"
        f"- Optimised for **{objective}** across **{len(scen)} stochastic races** "
        f"(safety cars, traffic, reliability) using common random numbers.\n"
        f"- Pit window placed where fresh-tyre pace gain overtakes the {prof.pit_loss_s:.0f}s pit loss."
    )

st.subheader("Candidate strategies")
st.dataframe(res.table(8), use_container_width=True)

# Surrogate heatmap -----------------------------------------------------------
st.subheader("Instant what-if — pit-lap × compound (learned surrogate)")
if net is None:
    st.info("Surrogate model not built yet. Run `python scripts/build_surrogate.py` to enable "
            "the millisecond full-space sweep. Showing nothing for now.")
else:
    laps, labels, Z, dt, n = heatmap_surrogate(net, model, grid, delta)
    fig = go.Figure(go.Heatmap(z=Z, x=laps, y=labels, colorscale="RdYlGn_r",
                               colorbar=dict(title="E[finish]")))
    fig.update_layout(xaxis_title="pit lap", yaxis_title="compound plan", height=340,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"⚡ Surrogate scored **{n}** strategies in **{dt:.0f} ms** "
               f"(~{1000*n/dt:.0f}/s) — a Monte-Carlo sweep of the same space would take minutes.")

# Risk frontier ---------------------------------------------------------------
st.subheader("Risk / reward frontier")
fr = res.frontier()
allc = res.ranked
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=[s.ensemble.mean_position for s in allc],
    y=[s.ensemble.cvar_position() for s in allc],
    mode="markers", marker=dict(size=8, color="lightgray"), name="candidates",
    text=[s.strategy.label() for s in allc]))
fig2.add_trace(go.Scatter(
    x=[s.ensemble.mean_position for s in fr],
    y=[s.ensemble.cvar_position() for s in fr],
    mode="markers+lines", marker=dict(size=11, color="crimson"), name="efficient frontier",
    text=[s.strategy.label() for s in fr]))
fig2.update_layout(xaxis_title="expected finish (lower better)",
                   yaxis_title="tail risk CVaR10 (lower better)", height=340,
                   margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

# Safety-car counterfactual ---------------------------------------------------
st.subheader("Safety-car counterfactual")
from pitwall.live import advise  # noqa: E402
from pitwall.live.updater import TyreBelief  # noqa: E402

mid = prof.n_laps // 2
belief = TyreBelief(model.pace.base, 0.08, np.diag([0.5, 0.0004]))
green = advise(belief, current_lap=mid, current_age=mid // 2, n_laps=prof.n_laps,
               pit_loss=model.pit.effective_loss("green"))
# Under SC the cheap stop is only available for the next ~3 laps (the window).
under_sc = advise(belief, current_lap=mid, current_age=mid // 2, n_laps=prof.n_laps,
                  pit_loss=model.pit.effective_loss("green"),
                  cheap_until=mid + 3, cheap_pit_loss=model.pit.effective_loss("SC"))
sc1, sc2 = st.columns(2)
sc1.metric("Optimal pit lap — GREEN", green.optimal_pit_lap,
           help=f"window {green.window_lo}-{green.window_hi}")
sc2.metric("Optimal pit lap — SC NOW", under_sc.optimal_pit_lap,
           delta=f"{under_sc.optimal_pit_lap - green.optimal_pit_lap} laps",
           help=f"window {under_sc.window_lo}-{under_sc.window_hi}")
st.caption(f"A safety car cuts the effective pit loss from {model.pit.effective_loss('green'):.0f}s "
           f"to {model.pit.effective_loss('SC'):.0f}s — the engine instantly re-evaluates and "
           f"pulls the optimal stop forward to grab the cheap stop.")
