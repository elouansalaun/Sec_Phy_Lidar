"""
src/analysis.py: where the sensors are, what they see, where the greedy loses.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

from src.instance import load_instance, non_detection

INK, INK_2 = "#0b0b0b", "#52514e"
C_OPT, C_GREEDY, NEUTRAL = "#2a78d6", "#eb6834", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#9cc3ef", C_OPT, "#12396b"])
DIV = LinearSegmentedColormap.from_list("div", [C_GREEDY, NEUTRAL, C_OPT])
WEAK_THRESHOLD = 0.5


# =============================================================================
# Step 1: loading
# =============================================================================
def hillshade(z: np.ndarray, az: float = 315, alt: float = 45) -> np.ndarray:
    gy, gx = np.gradient(z)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, e = np.radians(az), np.radians(alt)
    return np.clip(np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def solutions(path) -> dict[float, np.ndarray]:
    s = pd.read_csv(path)
    return {float(b): g.config.to_numpy() for b, g in s.groupby("budget")}


def to_grid(values: np.ndarray, rows, cols, shape) -> np.ndarray:
    g = np.full(shape, np.nan)
    g[rows, cols] = values
    return g


# =============================================================================
# Step 2: drawing the sensors
# =============================================================================
def draw_sensors(ax, T, configs, cand, hfov, r_wedge=22.0):
    sel = configs.iloc[T].merge(cand[["x", "y"]], left_on="cand", right_index=True)
    for _, c in sel.iterrows():
        if c.sensor in hfov:                           # camera: oriented sector
            t0 = 90 - c.azimuth - hfov[c.sensor] / 2   # clockwise azimuth from north -> trigonometric
            ax.add_patch(Wedge((c.x, c.y), r_wedge, t0, t0 + hfov[c.sensor], fc="none",
                               ec=INK, lw=1.1, zorder=5))
    for (cd, h), g in sel.groupby(["cand", "h"]):
        x, y = g.x.iloc[0], g.y.iloc[0]
        radar = (~g.sensor.isin(list(hfov))).any()
        ax.plot(x, y, marker="o" if radar else "s", ms=8 if radar else 5, mfc="white" if radar else INK,
                mec=INK, mew=1.6, zorder=6)
        ax.annotate(f"{h:g} m", (x, y), xytext=(5, -9), textcoords="offset points", fontsize=7,
                    color=INK, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def base_map(ax, hs, extent, site_layers):
    ax.imshow(hs, extent=extent, cmap="Greys_r", vmin=-0.6, vmax=1.3, zorder=0)
    site_layers["assets"].boundary.plot(ax=ax, color=INK, lw=0.8, zorder=4)
    site_layers["perimeter"].boundary.plot(ax=ax, color=INK_2, lw=1.2, ls="--", zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# =============================================================================
# Step 3: orchestration
# =============================================================================
def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    out = ROOT / cfg["optimization"]["out_dir"]
    figs = ROOT / cfg["optimization"]["figures_dir"]
    figs.mkdir(parents=True, exist_ok=True)
    budgets = [float(b) for b in cfg["optimization"].get("analysis_budgets", [4, 13, 30])]
    inst, configs = load_instance(cfg, ROOT)
    cel = np.load(ROOT / cfg["coverage"]["out_dir"] / "cells.npz")
    rows, cols, zone, w = cel["rows"], cel["cols"], cel["zone"], cel["w"]
    cover = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="coverage")
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"]).set_index("id")
    cand["x"], cand["y"] = cand.geometry.x, cand.geometry.y
    site_layers = {"assets": gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="assets"),
                   "perimeter": gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter")}
    hfov = {c["name"]: c["hfov_deg"] for c in cfg["sensors"]["cameras"]}
    sol_m, sol_g = solutions(out / "milp_solutions.csv"), solutions(out / "greedy_solutions.csv")

    with rasterio.open(ROOT / cfg["rasters"]["out_dir"] / "dsm.tif") as src:
        dsm, tr = src.read(1).astype(float), src.transform
    m = 25                                             # margin around the cells (px)
    r0, r1 = rows.min() - m, rows.max() + m + 1
    c0, c1 = cols.min() - m, cols.max() + m + 1
    hs = hillshade(np.nan_to_num(dsm[r0:r1, c0:c1], nan=np.nanmedian(dsm)))
    extent = (tr.c + c0 * tr.a, tr.c + c1 * tr.a, tr.f + r1 * tr.e, tr.f + r0 * tr.e)
    shape = dsm.shape

    det = {}
    for B in budgets:
        det[("milp", B)] = 1 - non_detection(inst, sol_m.get(B, []))
        det[("greedy", B)] = 1 - non_detection(inst, sol_g.get(B, []))

    # --- Figure 1: optimal solutions ----------------------------------------------------
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.6 * len(budgets), 4.3), dpi=150,
                             layout="constrained")
    for ax, B in zip(np.atleast_1d(axes), budgets):
        base_map(ax, hs, extent, site_layers)
        im = ax.imshow(to_grid(det[("milp", B)], rows, cols, shape)[r0:r1, c0:c1],
                       extent=extent, cmap=SEQ, vmin=0, vmax=1, zorder=2, interpolation="nearest")
        draw_sensors(ax, sol_m[B], configs, cand, hfov)
        cov = w @ det[("milp", B)] / w.sum()
        ax.set_title(f"B = {B:g}: coverage {100 * cov:.1f} %", loc="left", fontsize=11,
                     color=INK)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("P(detection) of the intruder", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    leg = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.6, ms=8, label="Radar"),
           Line2D([], [], marker="s", ls="", mfc=INK, mec=INK, ms=5, label="Camera support"),
           Line2D([], [], color=INK, lw=0.8, label="Assets"),
           Line2D([], [], color=INK_2, lw=1.2, ls="--", label="Fence")]
    if not cfg["sensors"]["radars"]:                   # cameras only
        leg = leg[1:]
    fig.legend(handles=leg, loc="outside lower center", ncol=len(leg), frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("Certified optimal solutions", x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(figs / "solutions_map.png", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: where the greedy loses -----------------------------------------------
    loss = {B: w @ (det[("milp", B)] - det[("greedy", B)]) / w.sum() for B in budgets}
    worst = [B for B in budgets if loss[B] > 1e-3]
    if worst:
        fig, axes = plt.subplots(1, len(worst), figsize=(4.6 * len(worst), 4.3), dpi=150,
                                 layout="constrained")
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        for ax, B in zip(np.atleast_1d(axes), worst):
            base_map(ax, hs, extent, site_layers)
            d = det[("milp", B)] - det[("greedy", B)]
            im = ax.imshow(to_grid(d, rows, cols, shape)[r0:r1, c0:c1], extent=extent, cmap=DIV,
                           norm=norm, zorder=2, interpolation="nearest")
            draw_sensors(ax, sol_g[B], configs, cand, hfov)
            ax.set_title(f"B = {B:g}: the greedy loses {100 * loss[B]:.1f} points",
                         loc="left", fontsize=11, color=INK)
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
        cb = fig.colorbar(im, ax=axes, shrink=0.85)
        cb.set_label("ΔP = P_optimum − P_greedy", color=INK_2)
        cb.ax.tick_params(colors=INK_2, labelsize=8)
        fig.suptitle("What the greedy misses. Blue: the optimum sees better; sensors of the greedy",
                     x=0.01, ha="left", fontsize=13, color=INK)
        fig.savefig(figs / "greedy_map.png", bbox_inches="tight")
        plt.close(fig)

    # --- Table per zone -----------------------------------------------------------------
    src_z = cover["source"].to_numpy()[zone]
    rows_t = []
    for z in cover["source"]:
        k = src_z == z
        r = dict(zone=z, zone_weight=cover.set_index("source").weight[z],
                 W_share=w[k].sum() / w.sum(), n_cells=int(k.sum()))
        for B in budgets:
            r[f"opt_{B:g}"] = w[k] @ det[("milp", B)][k] / w[k].sum()
            r[f"greedy_{B:g}"] = w[k] @ det[("greedy", B)][k] / w[k].sum()
            r[f"weak_{B:g}"] = (det[("milp", B)][k] < WEAK_THRESHOLD).mean()
        rows_t.append(r)
    zt = pd.DataFrame(rows_t)
    zt.to_csv(out / "zones.csv", index=False)
    pd.set_option("display.width", 220)
    print("Weighted coverage per zone (certified optimum) and share of cells with P < 0.5:\n")
    show = ["zone", "zone_weight", "W_share"] + [f"{p}_{B:g}" for B in budgets
                                                 for p in ("opt", "weak")]
    print(zt[show].round(3).to_string(index=False))
    print("\nGreedy loss per zone (coverage points, optimum − greedy):\n")
    print(pd.DataFrame({f"B={B:g}": zt[f"opt_{B:g}"] - zt[f"greedy_{B:g}"] for B in budgets})
            .set_index(zt.zone).round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
