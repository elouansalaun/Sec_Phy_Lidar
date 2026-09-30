"""
src/timely.py: timely detection (simplified EASI, docs/formulation.md §9.1).

A detection at v is only useful if the intruder still needs at least T_R seconds (response
time) to reach the target. We write d(v) for the minimal time from v to the target:

  d(v) = min_{γ : v → V_t} Σ_{e ∈ γ} (ℓ_e / speed + t_×(e))

d does not depend on the sensors: one multi-source Dijkstra from the targets, per profile.
Masked timely detection:  P̃_v = P_v · 1[d(v) ≥ T_R],  σ̃_v = σ_v · 1[d(v) ≥ T_R].

  p̃* = p*(P̃) is a lower bound on the guaranteed timely detection: along a path, the time
        really left at u is ≥ d(u), so every node that is useful after masking really is.
  Upper bound: real timely detection (remaining time measured along the path) on two
        feasible corridors of G_{p̃*} (the shortest in metres, the fastest).
  Proposition: if T_R > T_min = min_{s ∈ V_s} d(s), then p̃* = 0 for every layout.

Outputs : intruder/time_remaining.npz, intruder/timely.csv, intruder/timely_stats.csv,
          figures/time_remaining.png, figures/timely.png
Usage   : python -m src.timely
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from src.intruder import (PROFILES, Domain, _shortest, adjacency, corridor_threshold, crossing,
                          detection, edge_times, exposure)


# =============================================================================
# Step 1: remaining time d(v), independent of the sensors
# =============================================================================
def time_remaining(dom: Domain, ic: dict, profile: str) -> np.ndarray:
    """d(v) in seconds (inf if the target cannot be reached from v for this profile)."""
    mask = dom.edges(profile)
    M = adjacency(dom, mask, edge_times(dom, ic)[mask])
    return dijkstra(M, directed=False, indices=np.flatnonzero(dom.target), min_only=True)


def fastest_path(dom: Domain, ic: dict, profile: str, node_ok=None) -> np.ndarray:
    """Fastest path from the starts to the target (restricted to the nodes node_ok)."""
    mask = dom.edges(profile)
    if node_ok is not None:
        mask = mask & node_ok[dom.u] & node_ok[dom.v]
    return _shortest(dom, mask, edge_times(dom, ic)[mask], node_ok)


# =============================================================================
# Step 2: timely mask and real value along a path
# =============================================================================
def mask_timely(P: np.ndarray, sigma: np.ndarray, d: np.ndarray, T_R: float):
    useful = d >= T_R
    return np.where(useful, P, 0.0), np.where(useful, sigma, 0.0)


def remaining_along(dom: Domain, ic: dict, profile: str, path: np.ndarray) -> np.ndarray:
    """left[k] = real time from path[k] to the end of the path (0 at the last node)."""
    if path.size == 0:
        return np.zeros(0)
    mask = dom.edges(profile)
    Tm = adjacency(dom, mask, edge_times(dom, ic)[mask])
    t = np.asarray(Tm[path[:-1], path[1:]]).ravel()
    assert (t > 0).all(), "edge missing from the profile along the path"
    return np.r_[np.cumsum(t[::-1])[::-1], 0.0]


def path_value(P: np.ndarray, path: np.ndarray, left: np.ndarray, T_R: float) -> float:
    """Real timely detection at the best point of the path: max P over the nodes where left ≥ T_R."""
    useful = left >= T_R
    return float(P[path[useful]].max()) if useful.any() else 0.0


# =============================================================================
# Step 3: evaluation of a layout for a response time
# =============================================================================
def evaluate_timely(dom: Domain, P, sigma, ic: dict, T_R: float, d: dict) -> dict:
    """p̃*, upper bound and timely P_path (nominal τ_obs), for F and G."""
    res = {}
    for pr in PROFILES:
        Pm, sm = mask_timely(P, sigma, d[pr], T_R)
        p, corridor = corridor_threshold(dom, Pm, pr)
        fast = fastest_path(dom, ic, pr, Pm <= p)
        ub = min(path_value(P, c, remaining_along(dom, ic, pr, c), T_R)
                 for c in (corridor, fast) if c.size)
        assert p <= ub + 1e-12, "lower bound > upper bound"
        lam, _ = exposure(dom, sm, pr, ic, ic["tau_obs_s"])
        res[f"p_tilde_{pr}"] = p
        res[f"p_tilde_ub_{pr}"] = ub
        res[f"P_path_timely_{pr}"] = float(max(-np.expm1(-lam), 0.0))
    return res


def ceiling(A) -> tuple[np.ndarray, np.ndarray]:
    """P and σ with every configuration active: upper bound of any layout (P grows with T)."""
    L = A.tocsc(copy=True).astype(np.float64)
    with np.errstate(divide="ignore"):
        L.data = -np.log1p(-L.data)
    sigma = np.asarray(L.sum(axis=0)).ravel()
    return np.maximum(-np.expm1(-sigma), 0.0), sigma


# =============================================================================
# Step 4: reference solutions: NSGA-II front of week 4
# =============================================================================
def front_solutions(path, budgets) -> dict[float, tuple[np.ndarray, pd.Series]]:
    """For each B: solution of cost ≤ B with the best p*_F, then the best F."""
    f = pd.read_csv(path)
    out = {}
    for B in budgets:
        s = f[f["cost"] <= B + 1e-9].sort_values(["p_star_F", "F"], ascending=False).iloc[0]
        out[float(B)] = (np.array(str(s["configs"]).split(), dtype=np.int64), s)
    return out


# =============================================================================
# Step 5: figures
# =============================================================================
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
C_F, C_G = "#eb6834", "#4a3aa7"
C_SOL = ("#2a78d6", "#1f9e6e", "#b8456a")


def _frame(ax):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_time_remaining(dom, grid, per, assets, d, fastest, levels, path):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r, c = dom.rows, dom.cols
    r0, r1, c0, c1 = r.min() - 3, r.max() + 4, c.min() - 3, c.max() + 4
    ext = (grid.xmin + c0 * grid.res, grid.xmin + c1 * grid.res,
           grid.ymax - r1 * grid.res, grid.ymax - r0 * grid.res)
    X = grid.xmin + (c + 0.5) * grid.res
    Y = grid.ymax - (r + 0.5) * grid.res
    xs = grid.xmin + (np.arange(c0, c1) + 0.5) * grid.res
    ys = grid.ymax - (np.arange(r0, r1) + 0.5) * grid.res

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), dpi=150, layout="constrained")
    vmax = max(np.nanmax(np.where(np.isfinite(d[p]), d[p], np.nan)) for p in PROFILES)
    for ax, pr, col in zip(axes, PROFILES, (C_F, C_G)):
        img = np.full((grid.height, grid.width), np.nan)
        img[r, c] = np.where(np.isfinite(d[pr]), d[pr], np.nan)
        sub = img[r0:r1, c0:c1]
        im = ax.imshow(sub, extent=ext, cmap="viridis_r", vmin=0, vmax=vmax,
                       interpolation="nearest")
        cs = ax.contour(xs, ys, np.ma.masked_invalid(sub), levels=levels, colors="white",
                        linewidths=0.7)
        ax.clabel(cs, fmt=lambda v: f"{v:g} s", fontsize=7, colors="white")
        gpd.GeoSeries(list(assets.geometry)).plot(ax=ax, color="#d6d4cf", edgecolor=INK_2, lw=0.4)
        gpd.GeoSeries([per.boundary]).plot(ax=ax, color=INK, lw=1)
        p = fastest[pr]
        ax.plot(X[p], Y[p], color="white", lw=3.2)
        ax.plot(X[p], Y[p], color=col, lw=1.8)
        name = "determined (F)" if pr == "F" else "opportunistic (G)"
        ax.set_title(f"Intruder {name}\nT_min = {d[pr][dom.source].min():.1f} s",
                     loc="left", fontsize=10, color=INK)
        _frame(ax)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("Time left to reach the target d(v) (s)", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_curves(df, T_min, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), dpi=150, sharey=True,
                             layout="constrained")
    for ax, pr in zip(axes, PROFILES):
        for col, (B, g) in zip(C_SOL + ("#8a8883",), df.groupby("budget")):
            lab = "full catalog (ceiling)" if not np.isfinite(B) else f"B = {B:g}"
            ax.plot(g["T_R"], g[f"p_tilde_{pr}"], color=col, lw=2, label=lab,
                    zorder=1 if not np.isfinite(B) else 2)
            if np.isfinite(B):
                ax.plot(g["T_R"], g[f"p_tilde_ub_{pr}"], color=col, lw=1, ls="--")
        ax.axvline(T_min[pr], color=INK_2, lw=0.8, ls=":")
        ax.text(T_min[pr], 1.02, f" T_min = {T_min[pr]:.1f} s", color=INK_2, fontsize=8,
                va="bottom")
        name = "determined (F)" if pr == "F" else "opportunistic (G)"
        ax.set_title(f"Intruder {name}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Response time T_R (s)", color=INK_2)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, 60)                 # beyond: p̃* = 0 everywhere (proposition §9.1)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=INK_2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Guaranteed timely detection p̃*", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, title="solid: p̃*, dashed: upper bound",
                   title_fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Step 6: main program
# =============================================================================
def main() -> None:
    import geopandas as gpd

    from src.intruder import load_domain
    from src.rasters import load_grid
    from src.site import ROOT, load_config

    cfg = load_config()
    ic = cfg["intruder"]
    out = ROOT / ic["out_dir"]
    fig_dir = ROOT / cfg["optimization"]["figures_dir"]
    dom, A, conf = load_domain(cfg, ROOT)

    # 1. Remaining time and minimal intrusion time
    d = {pr: time_remaining(dom, ic, pr) for pr in PROFILES}
    assert (d["F"] <= d["G"] + 1e-9).all(), "violation of d_F ≤ d_G"
    np.savez_compressed(out / "time_remaining.npz", d_F=d["F"], d_G=d["G"])
    stats, fastest, T_min = [], {}, {}
    for pr in PROFILES:
        ds = d[pr][dom.source]
        ds = ds[np.isfinite(ds)]
        ch = fastest_path(dom, ic, pr)
        fastest[pr], T_min[pr] = ch, float(ds.min())
        node, gate = crossing(dom, ch)
        k = int(np.flatnonzero(ch == node)[0]) if node >= 0 else -1
        left = remaining_along(dom, ic, pr, ch)
        stats.append(dict(profile=pr, T_min=T_min[pr], median=float(np.median(ds)),
                          p90=float(np.percentile(ds, 90)), T_max=float(ds.max()),
                          length_m=float(np.sum(np.hypot(np.diff(dom.rows[ch]),
                                                         np.diff(dom.cols[ch])))),
                          gate=gate,
                          time_outside_s=float(left[0] - left[k]) if k >= 0 else np.nan,
                          time_inside_s=float(left[k]) if k >= 0 else np.nan))
    st = pd.DataFrame(stats)
    st.to_csv(out / "timely_stats.csv", index=False)
    print("Intrusion time from the starts (s)")
    print(st.round(2).to_string(index=False))

    # 2. Front solutions against T_R
    budgets = (13, 20, 30)
    sols = front_solutions(ROOT / cfg["nsga"]["out_dir"] / "front_global.csv", budgets)
    grid_TR = np.union1d(np.arange(0.0, 121.0, 2.5), ic["t_response_s"])
    rows = []
    cases = [(B, s["cost"], s["F"], detection(A, T)) for B, (T, s) in sols.items()]
    cases.append((np.inf, np.nan, np.nan, ceiling(A)))        # full catalog: ceiling
    for B, cost_, F, (P, sigma) in cases:
        for T_R in grid_TR:
            r = evaluate_timely(dom, P, sigma, ic, float(T_R), d)
            rows.append(dict(budget=B, cost=cost_, F=F, T_R=float(T_R), **r))
    df = pd.DataFrame(rows)
    for pr in PROFILES:            # proposition §9.1: T_R > T_min ⇒ p̃* = 0
        assert (df.loc[df["T_R"] > T_min[pr], f"p_tilde_{pr}"] == 0).all()
        r0 = df[(df["T_R"] == 0) & np.isfinite(df["budget"])]   # T_R = 0 gives back p* (week 4)
        ref = np.array([sols[B][1][f"p_star_{pr}"] for B in r0["budget"]])
        assert np.allclose(r0[f"p_tilde_{pr}"], ref, atol=1e-6), "T_R = 0 ≠ p* of the front"
    df.to_csv(out / "timely.csv", index=False)

    tab = df[df["T_R"].isin(ic["t_response_s"])]
    cols = ["budget", "T_R"] + [f"{m}_{pr}" for pr in PROFILES
                                for m in ("p_tilde", "p_tilde_ub", "P_path_timely")]
    pd.set_option("display.width", 200)
    print("\nTimely detection of the front solutions (best p*_F at cost ≤ B)")
    print(tab[cols].round(3).to_string(index=False))

    # 3. Figures
    g = load_grid(ROOT / cfg["site"]["grid_yaml"])
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="assets")
    levels = [t for t in ic["t_response_s"] if t > 0]
    plot_time_remaining(dom, g, per, assets, d, fastest, levels, fig_dir / "time_remaining.png")
    plot_curves(df, T_min, fig_dir / "timely.png")
    print(f"\n-> {fig_dir / 'time_remaining.png'}\n-> {fig_dir / 'timely.png'}")


if __name__ == "__main__":
    main()
