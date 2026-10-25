
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.demo import (best_of_front, certified_solutions, configurations_at, describe_layout,
                      evaluate_layout, layout_map, load_demo)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="LiDAR surveillance", page_icon="📡", layout="wide")


@st.cache_resource(show_spinner="Loading the site, the graph and the detection matrix...")
def data():
    return load_demo(ROOT)


D = data()
TARGET = D.assets.loc[D.assets["intruder_target"].astype(bool), "name"].iloc[0]
S = certified_solutions(D)

# =============================================================================
# Sidebar: the intruder, then the layout
# =============================================================================
sb = st.sidebar
sb.caption("**Fictitious** site overlaid on an arbitrary urban area. "
           "LiDAR data © Gouvernement du Québec (MRNF), CC BY 4.0.")

sb.header("Intruder")
T_R = sb.slider("Response time T_R (s)", 0, 150, 10, 5,
                help="A detection only counts if the intruder still needs at least T_R seconds "
                     f"to reach {TARGET} (timely detection, EASI model).")
speed = sb.select_slider("Speed (m/s)", [1.0, 1.5, 3.0], 1.5,
                         format_func=lambda v: {1.0: "1.0 slow", 1.5: "1.5 walking",
                                                3.0: "3.0 running"}[v])
t_c = sb.slider("Delay: fence climb (s)", 0, 120, 10, 5)
t_p = sb.slider("Delay: gate passage (s)", 0, 120, 3, 1)
ic = dict(D.cfg["intruder"], speed_m_s=float(speed), t_fence_s=float(t_c),
          t_gate_s=float(t_p))

sb.header("Layout")
mode = sb.radio("Source", ["Certified optimum", "NSGA-II front", "Manual placement"],
                help="Certified optimum: exact adversarial loop (week 7). "
                     "NSGA-II front: coverage / worst case (week 4). "
                     "Manual placement: add or remove sensors, the intruder answers.")

def _default(items, value):
    """Index of value in items (otherwise the last one): the loop may not cover every case."""
    return items.index(value) if value in items else len(items) - 1


if "T" not in st.session_state:
    d = S[(S.theta == 0.9) & (S.T_R == 10) & (S.fusion == "max")]
    st.session_state["T"] = d.iloc[0]["T"] if len(d) else np.zeros(0, np.int64)

if mode == "Certified optimum":
    sm = S[S["fusion"] == "max"]
    if sm.empty:
        sb.warning("No certified optimum for this site (see the log of src.adversarial_loop).")
        st.session_state["T"] = np.zeros(0, np.int64)
    else:
        thetas = sorted(sm["theta"].unique())
        th = sb.selectbox("Required timely detection θ", thetas, index=_default(thetas, 0.9))
        trs = sorted(sm[sm.theta == th]["T_R"].unique())
        tr = sb.selectbox("For a response time of (s)", trs, index=_default(trs, 10.0),
                          format_func=lambda v: f"{v:g}")
        row = sm[(sm.theta == th) & (sm.T_R == tr)].iloc[0]
        st.session_state["T"] = row["T"]
        sb.info(f"**Certified** minimal cost: {row['cost']:.1f} units.")
elif mode == "NSGA-II front":
    B = sb.slider("Budget", 2.0, 30.0, 16.0, 0.5)
    crit = sb.radio("Criterion", ["F", "p*_F"], horizontal=True,
                    help="F: weighted coverage. p*_F: guaranteed detection on the worst corridor.")
    s = best_of_front(D, B, "F" if crit == "F" else "p")
    if s is None:
        sb.warning("No front solution under this budget.")
        st.session_state["T"] = np.zeros(0, np.int64)
    else:
        st.session_state["T"] = s["T"]
else:
    T = list(map(int, st.session_state["T"]))
    with sb.expander("Add a sensor", expanded=True):
        cands = sorted(D.conf["cand"].unique())
        cand = st.selectbox("Location", cands)
        hs = sorted(D.conf.loc[D.conf["cand"] == cand, "h"].unique())
        h = st.selectbox("Height (m)", hs, format_func=lambda v: f"{v:g}")
        sensors = sorted(D.conf.loc[(D.conf["cand"] == cand) & np.isclose(D.conf["h"], h),
                                    "sensor"].unique())
        sensor = st.selectbox("Sensor", sensors)
        opts = configurations_at(D, cand, h, sensor)
        if sensor.startswith("radar"):
            j = st.selectbox("Tilt", opts.index,
                             format_func=lambda i: f"{D.conf.at[i, 'tilt']:g}°")
        else:
            j = st.selectbox("Azimuth", opts.index,
                             format_func=lambda i: f"{D.conf.at[i, 'azimuth']:g}°")
        if st.button("Add", use_container_width=True):
            T = sorted(set(T) | {int(j)})
    lab = {int(j): f"{D.conf.at[j, 'cand']} · {D.conf.at[j, 'sensor']} · h {D.conf.at[j, 'h']:g}"
           for j in T}
    remove = sb.multiselect("Remove", T, format_func=lambda j: lab[j])
    if sb.button("Remove the selection", use_container_width=True) and remove:
        T = [j for j in T if j not in remove]
    if sb.button("Clear all", use_container_width=True):
        T = []
    st.session_state["T"] = np.array(T, np.int64)

# =============================================================================
# Main page
# =============================================================================
st.title("Optimal surveillance of a site from LiDAR data")
tab_demo, tab_res, tab_meth = st.tabs(["Demo", "Key results", "Method"])

with tab_demo:
    r = evaluate_layout(D, st.session_state["T"], ic, float(T_R))
    k = st.columns(6)
    k[0].metric("Cost", f"{r['cost']:.1f}")
    k[1].metric("Coverage F", f"{100 * r['F']:.1f} %")
    k[2].metric("p* determined", f"{r['p_star_F']:.2f}",
                help="Guaranteed detection on the least watched corridor (intruder who climbs "
                     "the fence anywhere).")
    k[3].metric(f"p̃* at T_R = {T_R} s", f"{r['p_tilde_F']:.2f}",
                help="Same, counting only detections early enough for the response.")
    k[4].metric("p* opportunistic", f"{r['p_star_G']:.2f}", help="Intruder who uses a gate.")
    k[5].metric("T_min", f"{r['T_min_F']:.0f} s",
                help="Minimal intrusion time: beyond it, no layout can detect in time.")

    if T_R >= r["T_min_F"]:
        st.error(f"T_R = {T_R} s exceeds the minimal intrusion time ({r['T_min_F']:.0f} s): "
                 "**no layout** can guarantee a timely detection (proposition 2). "
                 "Only a larger delay (fence **and** gates) can make this case possible.")
    elif r["p_tilde_F"] < 0.5:
        st.warning("The determined intruder finds a corridor where they are not detected in time "
                   "(dotted line).")
    for v in r["violations"]:
        st.warning(f"Violated constraint: {v}")

    zone = st.toggle("Show the area where a detection comes too late", value=True)
    st.plotly_chart(layout_map(D, r, float(T_R), show_dead_zone=zone), use_container_width=True)
    with st.expander(f"Layout: {len(r['T'])} sensor(s)"):
        st.dataframe(describe_layout(D, r["T"]), hide_index=True, use_container_width=True)

with tab_res:
    fig = ROOT / D.cfg["optimization"]["figures_dir"]
    blocks = [
        ("time_remaining.png", "Detecting is not interrupting",
         "Time left to reach the target. Beyond the minimal intrusion time, no layout can detect "
         "in time, whatever the budget."),
        ("robust_cost.png", "Certified minimal cost",
         "Exact optimiser ↔ intruder loop: each point is a proven optimum."),
        ("delay_frontier.png", "Delay against response time",
         "Minimal cost of a timely detection when the fence and the gates delay the intruder."),
        ("regret.png", "Sensitivity by decision regret",
         "What the nominal decisions cost when an assumption is wrong."),
        ("decoupling_paths.png", "Covering is not protecting",
         "At equal budget, the coverage optimum against the best worst-case layout."),
    ]
    for f, title, text in blocks:
        path = next((d / f for d in (fig, ROOT / D.cfg["intruder"]["out_dir"]) if (d / f).exists()),
                    None)
        if path is not None:
            st.subheader(title)
            st.caption(text)
            st.image(str(path), use_container_width=True)

with tab_meth:
    st.markdown("""
**Full chain** (details in the README and `docs/formulation.md`):

1. **LiDAR → rasters**: DTM, buildings, vegetation height and gap fraction (Beer-Lambert law).
2. **Exact visibility**: traversal of every crossed cell, opaque buildings, semi-transparent vegetation; validated against analytical solutions and an exact referee.
3. **Sensors**: camera (pixel density, EN 62676-4) and radar (Swerling I), or cameras only (`sensors.cameras_only`).
4. **Coverage optimisation**: greedy, certified optimum by submodular cuts, NSGA-II.
5. **The intruder**: least-detected corridor p\\* (union-find), timely detection p̃\\* (EASI).
6. **Exact adversarial loop**: constraint generation, the intruder acts as the oracle; certified optima.
7. **Sensitivity** by decision regret, preregistered protocol.

**Ethics.** The site, its zones and its assets are made up. Publishing blind-spot maps of a real sensitive site would be unacceptable.
""")
