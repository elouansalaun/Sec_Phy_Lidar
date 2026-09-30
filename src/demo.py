"""
src/demo.py: logic of the Streamlit demo (app.py), without any interface: testable alone.

Two uses:
  - browse the solutions already certified (src.adversarial_loop, src.nsga);
  - evaluate live a layout modified by hand: detection per cell, intruder corridors
    (p*, p̃*), remaining time d(v), cost, coverage F. One evaluation takes a few tens of
    milliseconds: the intruder "answers" every sensor added or removed.

Nothing is optimised here: the demo reuses the functions of the computation modules.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.adversarial_loop import front_nsga
from src.instance import cost, from_configs, violations
from src.intruder import PROFILES, corridor_threshold, detection, exposure, load_domain
from src.intruder_domain import config_key
from src.timely import time_remaining


# =============================================================================
# Step 1: loading (once, cached by the application)
# =============================================================================
@dataclass
class DemoData:
    cfg: dict
    dom: object
    A: object
    conf: pd.DataFrame
    inst: object
    keys: np.ndarray
    w: np.ndarray
    grid: object
    perimeter: object
    assets: object
    candidates: object
    background: np.ndarray      # RGB image of the area (hillshade + buildings)
    window: tuple               # (r0, r1, c0, c1): framing on the domain
    front: pd.DataFrame
    solutions: pd.DataFrame


def load_demo(root) -> DemoData:
    import geopandas as gpd

    from src.rasters import load_grid
    from src.viewshed import read
    from src.site import load_config

    cfg = load_config(root / "config.yaml")
    dom, A, conf = load_domain(cfg, root)
    cells = np.load(root / cfg["coverage"]["out_dir"] / "cells.npz")
    n_int = int(dom.inside.sum())
    inst = from_configs(A[:, :n_int], cells["w"], conf, cfg)
    grid = load_grid(root / cfg["site"]["grid_yaml"])
    per = gpd.read_file(root / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    assets = gpd.read_file(root / cfg["site"]["output_gpkg"], layer="assets")
    cand = gpd.read_file(root / cfg["candidates"]["output"])

    r, c = dom.rows, dom.cols
    win = (max(r.min() - 5, 0), min(r.max() + 6, grid.height),
           max(c.min() - 5, 0), min(c.max() + 6, grid.width))
    d = root / cfg["rasters"]["out_dir"]
    dtm, building = read(d / "dtm.tif"), read(d / "building_h.tif")
    background = background_image(dtm, building)[win[0]:win[1], win[2]:win[3]]

    front = pd.read_csv(root / cfg["nsga"]["out_dir"] / "front_global.csv")
    front["T"] = [np.array(str(s).split(), np.int64) for s in front["configs"]]
    front = front.drop_duplicates("configs").reset_index(drop=True)
    f_sol = root / cfg["loop"]["out_dir"] / "solutions.csv"
    try:                                # missing, or empty if no optimum was certified
        sol = pd.read_csv(f_sol)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        sol = pd.DataFrame(columns=["theta", "T_R", "fusion", "config"])
    return DemoData(cfg, dom, A, conf, inst, config_key(conf).to_numpy(), cells["w"], grid, per,
                    assets, cand, background, win, front, sol)


def background_image(dtm: np.ndarray, building: np.ndarray) -> np.ndarray:
    """Hillshade of the relief (sun in the north-west) and buildings in grey: neutral backdrop."""
    z = np.nan_to_num(dtm + np.nan_to_num(building), nan=float(np.nanmean(dtm)))
    gy, gx = np.gradient(z)
    az, alt = np.radians(315), np.radians(45)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    v = 0.80 + 0.15 * np.clip(shade, 0, 1)
    img = np.stack([v, v, v], axis=-1)
    img[np.nan_to_num(building) > 0] = (0.62, 0.61, 0.59)
    return (255 * img).astype(np.uint8)


# =============================================================================
# Step 2: evaluation of a layout
# =============================================================================
def evaluate_layout(D: DemoData, T, ic: dict, T_R: float) -> dict:
    """Everything the demo shows for a layout T and intruder parameters ic."""
    T = np.unique(np.asarray(T, np.int64))
    P, sigma = detection(D.A, T)
    n_int = int(D.dom.inside.sum())
    out = dict(T=T, P=P, cost=cost(D.inst, T) if T.size else 0.0,
               F=float(D.w @ P[:n_int]) / float(D.w.sum()),
               violations=violations(D.inst, T, 1e9) if T.size else [])
    for pr in PROFILES:
        d = time_remaining(D.dom, ic, pr)
        p, corridor = corridor_threshold(D.dom, P, pr)
        pt, corridor_t = corridor_threshold(D.dom, np.where(d >= T_R, P, 0.0), pr)
        lam, _ = exposure(D.dom, sigma, pr, ic, ic["tau_obs_s"])
        out.update({f"d_{pr}": d, f"T_min_{pr}": float(d[D.dom.source].min()),
                    f"p_star_{pr}": p, f"corridor_{pr}": corridor,
                    f"p_tilde_{pr}": pt, f"corridor_timely_{pr}": corridor_t,
                    f"P_path_{pr}": float(max(-np.expm1(-lam), 0.0))})
    return out


def describe_layout(D: DemoData, T) -> pd.DataFrame:
    """Readable table of a layout."""
    T = np.unique(np.asarray(T, np.int64))
    if T.size == 0:
        return pd.DataFrame(columns=["config", "location", "support", "h (m)", "sensor",
                                     "azimuth", "tilt", "sensor cost"])
    c = D.conf.iloc[T]
    return pd.DataFrame({
        "config": T, "location": c["cand"].to_numpy(), "support": c["support"].to_numpy(),
        "h (m)": c["h"].to_numpy(), "sensor": c["sensor"].to_numpy(),
        "azimuth": c["azimuth"].map(lambda a: "360°" if pd.isna(a) else f"{a:g}°").to_numpy(),
        "tilt": c["tilt"].map(lambda a: f"{a:.0f}°").to_numpy(),
        "sensor cost": c["sensor_cost"].to_numpy()})


# =============================================================================
# Step 3: reference solutions
# =============================================================================
def certified_solutions(D: DemoData) -> pd.DataFrame:
    """One row per certified optimum of src.adversarial_loop: θ, T_R, fusion, cost, configurations."""
    rows = []
    for (th, T_R, fu), g in D.solutions.groupby(["theta", "T_R", "fusion"]):
        T = g["config"].to_numpy(np.int64)
        rows.append(dict(theta=th, T_R=T_R, fusion=fu, cost=cost(D.inst, T), T=T))
    return pd.DataFrame(rows, columns=["theta", "T_R", "fusion", "cost", "T"])


def best_of_front(D: DemoData, budget: float, criterion: str) -> pd.Series | None:
    """Solution of the NSGA-II front with cost ≤ budget, best according to "F" or "p*_F"."""
    f = D.front[D.front["cost"] <= budget + 1e-9]
    if f.empty:
        return None
    order = ["F", "p_star_F"] if criterion == "F" else ["p_star_F", "F"]
    return f.sort_values(order, ascending=False).iloc[0]


def configurations_at(D: DemoData, cand: str, h: float, sensor: str) -> pd.DataFrame:
    """Configurations available for a location, a height and a sensor."""
    c = D.conf
    m = (c["cand"] == cand) & np.isclose(c["h"], h) & (c["sensor"] == sensor)
    return c[m]


# =============================================================================
# Step 4: map (plotly)
# =============================================================================
C_F, C_G, C_TIMELY, INK = "#eb6834", "#4a3aa7", "#b8456a", "#0b0b0b"


def layout_map(D: DemoData, res: dict, T_R: float, show_dead_zone: bool = True,
               timely_profile: str = "F", height: int = 640):
    import plotly.graph_objects as go
    import shapely

    g, (r0, r1, c0, c1) = D.grid, D.window
    x0, y0 = g.xmin + c0 * g.res, g.ymax - r0 * g.res

    def xy(rows, cols):
        return g.xmin + (cols + 0.5) * g.res - x0, g.ymax - (rows + 0.5) * g.res - y0

    fig = go.Figure()
    fig.add_trace(go.Image(z=D.background, x0=0.5, dx=g.res, y0=-0.5, dy=-g.res, hoverinfo="skip"))

    img = np.full((r1 - r0, c1 - c0), np.nan)
    img[D.dom.rows - r0, D.dom.cols - c0] = res["P"]
    fig.add_trace(go.Heatmap(z=img, x0=0.5, dx=g.res, y0=-0.5, dy=-g.res, zmin=0, zmax=1,
                             colorscale=[[0, "rgba(42,120,214,0.05)"], [1, "rgba(42,120,214,0.85)"]],
                             colorbar=dict(title=dict(text="P detection", font=dict(color=INK)),
                                           tickfont=dict(color=INK), thickness=12, len=0.6),
                             hovertemplate="P = %{z:.2f}<extra></extra>"))
    if show_dead_zone and T_R > 0:
        dz = np.full_like(img, np.nan)
        d = res[f"d_{timely_profile}"]
        dead = d < T_R
        dz[D.dom.rows[dead] - r0, D.dom.cols[dead] - c0] = 1.0
        fig.add_trace(go.Heatmap(z=dz, x0=0.5, dx=g.res, y0=-0.5, dy=-g.res, showscale=False,
                                 colorscale=[[0, "rgba(120,120,120,0.35)"],
                                             [1, "rgba(120,120,120,0.35)"]],
                                 hovertemplate="detection too late (d < T_R)<extra></extra>"))

    def line(geom, **kw):
        xs, ys = shapely.get_coordinates(geom).T
        fig.add_trace(go.Scatter(x=xs - x0, y=ys - y0, mode="lines", hoverinfo="skip",
                                 showlegend=False, **kw))

    for geom, name in zip(D.assets.geometry, D.assets["name"]):
        xs, ys = shapely.get_coordinates(geom.exterior).T
        fig.add_trace(go.Scatter(x=xs - x0, y=ys - y0, mode="lines", fill="toself",
                                 fillcolor="rgba(214,212,207,0.9)", line=dict(color="#52514e", width=1),
                                 name=name, hovertemplate=f"{name}<extra></extra>", showlegend=False))
    line(D.perimeter.exterior, line=dict(color=INK, width=2))

    for key, col, name, dash in (("corridor_G", C_G, "Opportunistic corridor (G)", "solid"),
                                 ("corridor_F", C_F, "Determined corridor (F)", "solid"),
                                 (f"corridor_timely_{timely_profile}", C_TIMELY,
                                  f"Timely corridor ({timely_profile}, T_R = {T_R:g} s)", "dot")):
        p = res[key]
        if key.startswith("corridor_timely") and T_R == 0:
            continue
        x, y = xy(D.dom.rows[p], D.dom.cols[p])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="white", width=5),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                                 line=dict(color=col, width=2.5, dash=dash), hoverinfo="skip"))

    T = res["T"]
    if T.size:
        c = D.conf.iloc[T].merge(D.candidates[["id", "geometry"]], left_on="cand", right_on="id")
        px = np.array([g_.x for g_ in c["geometry"]]) - x0
        py = np.array([g_.y for g_ in c["geometry"]]) - y0
        radar = c["sensor"].str.startswith("radar").to_numpy()
        txt = [f"{a} · {b} · h = {h:g} m" + ("" if pd.isna(z) else f" · azimuth {z:g}°")
               for a, b, h, z in zip(c["cand"], c["sensor"], c["h"], c["azimuth"])]
        fig.add_trace(go.Scatter(x=px[radar], y=py[radar], mode="markers", name="Radar",
                                 marker=dict(symbol="circle", size=13, color="#1f9e6e",
                                             line=dict(color="white", width=2)),
                                 text=np.array(txt)[radar], hovertemplate="%{text}<extra></extra>"))
        cam = ~radar
        if cam.any():
            az = np.radians(c["azimuth"].to_numpy()[cam].astype(float))
            L = 12.0                                    # line of sight arrow (azimuth from north)
            for xa, ya, a in zip(px[cam], py[cam], az):
                fig.add_trace(go.Scatter(x=[xa, xa + L * np.sin(a)], y=[ya, ya + L * np.cos(a)],
                                         mode="lines", line=dict(color="#0b0b0b", width=2),
                                         hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=px[cam], y=py[cam], mode="markers", name="Camera",
                                     marker=dict(symbol="square", size=10, color="#0b0b0b",
                                                 line=dict(color="white", width=2)),
                                     text=np.array(txt)[cam], hovertemplate="%{text}<extra></extra>"))

    W, H = (c1 - c0) * g.res, (r1 - r0) * g.res
    fig.update_xaxes(range=[0, W], visible=False, constrain="domain")
    fig.update_yaxes(range=[-H, 0], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.08, x=0,
                                  font=dict(size=11, color=INK)))
    return fig
