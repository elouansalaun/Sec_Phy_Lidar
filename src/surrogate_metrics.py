"""
src/surrogate_metrics.py: metrics shared by the baselines and the network.


"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.surrogate_dataset import STRATA, disc_bins


def bin_areas(cfg) -> np.ndarray:
    sc = cfg["surrogate"]
    *_, n_bin = disc_bins(sc["radius_m"], sc["rings_m"], sc["n_sectors"],
                          cfg["site"]["resolution_m"])
    return n_bin.reshape(len(sc["rings_m"]) - 1, sc["n_sectors"])


def total_potential(Y: np.ndarray, n_bin: np.ndarray) -> np.ndarray:
    """(n, h, ring, sector) -> (n, h): share of the whole disc that is visible ground."""
    return np.einsum("nhas,as->nh", Y, n_bin) / n_bin.sum()


def nms(score, x, y, k: int, radius: float) -> np.ndarray:
    """Indices of the k best scores, pairwise at least `radius` apart."""
    order = np.argsort(-score)
    taken = []
    for i in order:
        if all((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 >= radius ** 2 for j in taken):
            taken.append(i)
            if len(taken) == k:
                break
    return np.asarray(taken)


def decision(Vp, Vt, x, y, k=20, radius=10.0, tol=5.0) -> tuple[float, float]:
    ref = nms(Vt, x, y, k, radius)
    sel = nms(Vp, x, y, k, radius)
    d2 = (x[ref][:, None] - x[sel][None]) ** 2 + (y[ref][:, None] - y[sel][None]) ** 2
    recall = float(np.mean(d2.min(axis=1) <= tol ** 2))
    regret = float(1.0 - Vt[sel].sum() / Vt[ref].sum())
    return recall, regret


def report(Yp, Yt, stratum, heights, n_bin, inside=None, xy=None, name="") -> pd.DataFrame:
    """One row per height: MAE (global, per ring, per stratum), Spearman on V̄, decision."""
    Vp, Vt = total_potential(Yp, n_bin), total_potential(Yt, n_bin)
    err = np.abs(Yp - Yt)
    rows = []
    for ih, h in enumerate(heights):
        r = dict(model=name, h=h, MAE=err[:, ih].mean(),
                 MAE_0_50=err[:, ih, 0].mean(), MAE_50_150=err[:, ih, 1].mean())
        for k, s in enumerate(STRATA):
            m = stratum == k
            r[f"MAE_{s}"] = err[m, ih].mean() if m.any() else np.nan
        r["spearman_V"] = spearmanr(Vp[:, ih], Vt[:, ih])[0]
        if inside is not None:
            m = inside
            r["recall_top20"], r["regret_top20"] = decision(Vp[m, ih], Vt[m, ih],
                                                            xy[0][m], xy[1][m])
        rows.append(r)
    return pd.DataFrame(rows)


def site_interior(cfg, root, rows, cols):
    """Positions of the test block inside the perimeter, and their coordinates."""
    import geopandas as gpd
    import shapely
    import yaml

    g = yaml.safe_load(open(root / cfg["site"]["grid_yaml"]))
    x = g["xmin"] + (cols + 0.5) * g["res"]
    y = g["ymax"] - (rows + 0.5) * g["res"]
    per = gpd.read_file(root / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    return shapely.contains_xy(per, x, y), (x, y)


# =============================================================================
# Error against cost of every surrogate (reads baselines.csv and surrogate*.csv)
# =============================================================================
def plot_cost_error(root, cfg, path, h: float = 6.0, exact_us: float = 8800.0) -> pd.DataFrame:
    """exact_us: cost of the exact computation per position, pilot measure of
    `python -m src.surrogate_dataset --pilot 200` (the value of docs/journal.md). The full
    labelling run stored in labels.npz was slower (≈ 67 ms per position, machine under load)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root / cfg["surrogate"]["out_dir"]
    base = pd.read_csv(out / "baselines.csv")
    unet = pd.read_csv(out / "surrogate.csv")
    phys = pd.read_csv(out / "surrogate_physics.csv")
    at_h = lambda d: d[np.isclose(d.h, h)].iloc[0]
    pts = [("Raw U-Net", 1e3 * at_h(unet).cost_ms_position, at_h(unet).MAE, "^", "#eb6834"),
           ("Physics U-Net (horizons as input)", at_h(phys).cost_us_position, at_h(phys).MAE,
            "^", "#1baf7a"),
           ("B2: pyramid + boosting", 1e3 * at_h(base[base.model == "B2 (test)"]).cost_ms_position,
            at_h(base[base.model == "B2 (test)"]).MAE, "s", "#2a78d6"),
           ("B1: exact horizons + boosting", 1e3 * at_h(base[base.model == "B1 (test)"]).cost_ms_position,
            at_h(base[base.model == "B1 (test)"]).MAE, "s", "#4a3aa7"),
           ("Exact computation", exact_us, 0.0, "o", "#0b0b0b")]
    df = pd.DataFrame(pts, columns=["model", "cost_us", "MAE", "marker", "color"])
    front = df.sort_values("cost_us")
    front = front[front.MAE <= front.MAE.cummin()]

    INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    ax.plot(front.cost_us, 100 * front.MAE, color="#c9c8c3", lw=1.5, zorder=1,
            label="cost / error front (non-dominated models)")
    for _, r in df.iterrows():
        ax.scatter(r.cost_us, 100 * r.MAE, marker=r.marker, s=46, color=r.color, zorder=3,
                   edgecolor="white", lw=0.8)
        below = r.model.startswith("B2")
        ax.annotate(r.model, (r.cost_us, 100 * r.MAE), xytext=(8, -14 if below else 7),
                    textcoords="offset points", fontsize=8.5, color=INK_2,
                    ha="right" if r.model.startswith("Exact") else "left")
    ax.set_xscale("log")
    ax.set_xlim(0.08, 3e4)
    ax.set_ylim(-0.5, 100 * df.MAE.max() + 1.3)
    ax.set_xlabel("Cost per position (µs, MacBook Air; networks on MPS, the rest on CPU)")
    ax.set_ylabel("Mean absolute error on V (%)")
    ax.set_title(f"Surrogate models: error against cost (test block, h = {h:g} m)", loc="left",
                 fontsize=11, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return df


if __name__ == "__main__":
    from src.site import ROOT, load_config
    cfg = load_config()
    path = ROOT / cfg["optimization"]["figures_dir"] / "surrogate_cost_error.png"
    print(plot_cost_error(ROOT, cfg, path)[["model", "cost_us", "MAE"]].round(4).to_string(index=False))
    print(f"-> {path}")
