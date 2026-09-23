"""
src/analyse.py — Où sont les capteurs, que voient-ils, où le glouton perd-il (étape 3.5).

  - carte_solutions.png : P(détection) des solutions optimales, capteurs placés, pour 3 budgets
  - carte_glouton.png   : P_optimum − P_glouton par cellule, là où le glouton perd le plus
  - zones.csv           : couverture pondérée par zone, optimum et glouton, part de cellules faibles

Lit les sorties de src.coverage, src.greedy et src.milp, sans rien recalculer.
Usage : python -m src.analyse
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
C_OPT, C_GLOUTON, NEUTRE = "#2a78d6", "#eb6834", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#9cc3ef", C_OPT, "#12396b"])
DIV = LinearSegmentedColormap.from_list("div", [C_GLOUTON, NEUTRE, C_OPT])
SEUIL_FAIBLE = 0.5


# =============================================================================
# Étape 1 — Chargement
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
# Étape 2 — Dessin des capteurs
# =============================================================================
def draw_sensors(ax, T, configs, cand, hfov, r_wedge=22.0):
    sel = configs.iloc[T].merge(cand[["x", "y"]], left_on="cand", right_index=True)
    for _, c in sel.iterrows():
        if c.capteur in hfov:                          # caméra : secteur orienté
            t0 = 90 - c.azimut - hfov[c.capteur] / 2  # azimut horaire depuis le nord -> trigo
            ax.add_patch(Wedge((c.x, c.y), r_wedge, t0, t0 + hfov[c.capteur], fc="none",
                               ec=INK, lw=1.1, zorder=5))
    for (cd, h), g in sel.groupby(["cand", "h"]):
        x, y = g.x.iloc[0], g.y.iloc[0]
        radar = (~g.capteur.isin(list(hfov))).any()
        ax.plot(x, y, marker="o" if radar else "s", ms=8 if radar else 5, mfc="white" if radar else INK,
                mec=INK, mew=1.6, zorder=6)
        ax.annotate(f"{h:g} m", (x, y), xytext=(5, -9), textcoords="offset points", fontsize=7,
                    color=INK, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def base_map(ax, hs, extent, site_layers):
    ax.imshow(hs, extent=extent, cmap="Greys_r", vmin=-0.6, vmax=1.3, zorder=0)
    site_layers["actifs"].boundary.plot(ax=ax, color=INK, lw=0.8, zorder=4)
    site_layers["perimetre"].boundary.plot(ax=ax, color=INK_2, lw=1.2, ls="--", zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# =============================================================================
# Étape 3 — Orchestration
# =============================================================================
def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    out = ROOT / cfg["optimisation"]["out_dir"]
    budgets = [float(b) for b in cfg["optimisation"].get("analyse_budgets", [4, 13, 30])]
    inst, configs = load_instance(cfg, ROOT)
    cel = np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")
    rows, cols, zone, w = cel["rows"], cel["cols"], cel["zone"], cel["w"]
    cover = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="couverture")
    cand = gpd.read_file(ROOT / cfg["candidates"]["output"]).set_index("id")
    cand["x"], cand["y"] = cand.geometry.x, cand.geometry.y
    site_layers = {"actifs": gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="actifs"),
                   "perimetre": gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre")}
    hfov = {c["name"]: c["hfov_deg"] for c in cfg["sensors"]["cameras"]}
    sol_m, sol_g = solutions(out / "milp_solutions.csv"), solutions(out / "greedy_solutions.csv")

    with rasterio.open(ROOT / cfg["rasters"]["out_dir"] / "mns.tif") as src:
        mns, tr = src.read(1).astype(float), src.transform
    m = 25                                             # marge autour des cellules (px)
    r0, r1 = rows.min() - m, rows.max() + m + 1
    c0, c1 = cols.min() - m, cols.max() + m + 1
    hs = hillshade(np.nan_to_num(mns[r0:r1, c0:c1], nan=np.nanmedian(mns)))
    extent = (tr.c + c0 * tr.a, tr.c + c1 * tr.a, tr.f + r1 * tr.e, tr.f + r0 * tr.e)
    shape = mns.shape

    det = {}
    for B in budgets:
        det[("milp", B)] = 1 - non_detection(inst, sol_m.get(B, []))
        det[("glouton", B)] = 1 - non_detection(inst, sol_g.get(B, []))

    # --- Figure 1 : solutions optimales -----------------------------------------------------
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.6 * len(budgets), 4.3), dpi=150,
                             layout="constrained")
    for ax, B in zip(np.atleast_1d(axes), budgets):
        base_map(ax, hs, extent, site_layers)
        im = ax.imshow(to_grid(det[("milp", B)], rows, cols, shape)[r0:r1, c0:c1],
                       extent=extent, cmap=SEQ, vmin=0, vmax=1, zorder=2, interpolation="nearest")
        draw_sensors(ax, sol_m[B], configs, cand, hfov)
        cov = w @ det[("milp", B)] / w.sum()
        ax.set_title(f"B = {B:g}  —  couverture {100 * cov:.1f} %", loc="left", fontsize=11,
                     color=INK)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("P(détection) de l'intrus", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    leg = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.6, ms=8, label="Radar"),
           Line2D([], [], marker="s", ls="", mfc=INK, mec=INK, ms=5, label="Support caméras"),
           Line2D([], [], color=INK, lw=0.8, label="Actifs"),
           Line2D([], [], color=INK_2, lw=1.2, ls="--", label="Clôture")]
    fig.legend(handles=leg, loc="outside lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("Solutions optimales certifiées", x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(out / "carte_solutions.png", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2 : où le glouton perd -----------------------------------------------------
    perte = {B: w @ (det[("milp", B)] - det[("glouton", B)]) / w.sum() for B in budgets}
    worst = [B for B in budgets if perte[B] > 1e-3]
    if worst:
        fig, axes = plt.subplots(1, len(worst), figsize=(4.6 * len(worst), 4.3), dpi=150,
                                 layout="constrained")
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        for ax, B in zip(np.atleast_1d(axes), worst):
            base_map(ax, hs, extent, site_layers)
            d = det[("milp", B)] - det[("glouton", B)]
            im = ax.imshow(to_grid(d, rows, cols, shape)[r0:r1, c0:c1], extent=extent, cmap=DIV,
                           norm=norm, zorder=2, interpolation="nearest")
            draw_sensors(ax, sol_g[B], configs, cand, hfov)
            ax.set_title(f"B = {B:g}  —  le glouton perd {100 * perte[B]:.1f} points",
                         loc="left", fontsize=11, color=INK)
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
        cb = fig.colorbar(im, ax=axes, shrink=0.85)
        cb.set_label("ΔP = P_optimum − P_glouton", color=INK_2)
        cb.ax.tick_params(colors=INK_2, labelsize=8)
        fig.suptitle("Ce que le glouton ne voit pas — bleu : l'optimum voit mieux ; capteurs du glouton",
                     x=0.01, ha="left", fontsize=13, color=INK)
        fig.savefig(out / "carte_glouton.png", bbox_inches="tight")
        plt.close(fig)

    # --- Tableau par zone ------------------------------------------------------------------
    src_z = cover["source"].to_numpy()[zone]
    rows_t = []
    for z in cover["source"]:
        k = src_z == z
        r = dict(zone=z, poids_zone=cover.set_index("source").poids[z],
                 part_W=w[k].sum() / w.sum(), n_cellules=int(k.sum()))
        for B in budgets:
            r[f"opt_{B:g}"] = w[k] @ det[("milp", B)][k] / w[k].sum()
            r[f"glouton_{B:g}"] = w[k] @ det[("glouton", B)][k] / w[k].sum()
            r[f"faible_{B:g}"] = (det[("milp", B)][k] < SEUIL_FAIBLE).mean()
        rows_t.append(r)
    zt = pd.DataFrame(rows_t)
    zt.to_csv(out / "zones.csv", index=False)
    pd.set_option("display.width", 220)
    print("Couverture pondérée par zone (optimum certifié) et part de cellules à P < 0,5 :\n")
    show = ["zone", "poids_zone", "part_W"] + [f"{p}_{B:g}" for B in budgets
                                               for p in ("opt", "faible")]
    print(zt[show].round(3).to_string(index=False))
    print("\nPerte du glouton par zone (points de couverture, optimum − glouton) :\n")
    print(pd.DataFrame({f"B={B:g}": zt[f"opt_{B:g}"] - zt[f"glouton_{B:g}"] for B in budgets})
            .set_index(zt.zone).round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
