"""
src/sensitivity.py: sensitivity analysis by decision regret (docs/formulation.md §9.3).

Preregistered protocol (`sensitivity` block of config.yaml): we take the nominal decisions
D_0 and D_10 (certified optima of src.adversarial_loop, θ = 0.9, T_R = 0 or 10 s, max fusion)
and measure, for each scenario s (one factor at a time):

  holds       p̃*_s(D) ≥ θ
  c*_s        certified minimal cost under s (same loop)
  regret      cost(D) − c*_s if D holds (over-sizing)
  extra cost  c*_s − cost(D) if D does not hold (failure)
  Jaccard     between D and D*_s, configurations identified by their key

Factors: delay (t_c, t_p) and speed (only d changes); band b and foliage f
(domain and A_intruder rebuilt by the functions of src.intruder_domain and src.coverage;
k × f ⇔ α × f because τ is linear in k).
Delay-response frontier: T_min and c*(θ, T_R) for each (t_c, t_p).

Outputs : sensitivity/regret.csv, sensitivity/frontier.csv,
          figures/delay_frontier.png, figures/regret.png
Usage   : python -m src.sensitivity            (≈ 1 to 2 h on 2 cores)
          python -m src.sensitivity figures    (figures from the CSVs)
"""
from __future__ import annotations

import copy
import sys
import time

import numpy as np
import pandas as pd

from src.adversarial_loop import adversarial_loop, front_nsga, p_tilde
from src.instance import cost, from_configs
from src.intruder import Domain, load_domain
from src.intruder_domain import config_key
from src.timely import time_remaining

PROFILE = "F"


# =============================================================================
# Step 1: scenarios: rebuilding the domain and A_intruder
# =============================================================================
def build_scenario(cfg: dict, root, band: float, foliage: float):
    """(Domain, A, conf) for a band and a foliage factor; the nominal case is read from disk."""
    ic = cfg["intruder"]
    if np.isclose(band, ic["outer_band_m"]) and np.isclose(foliage, 1.0):
        return load_domain(cfg, root)

    import geopandas as gpd

    from src.coverage import build_matrix
    from src.intruder_domain import build_edges, domain_cells, gate_edges
    from src.viewshed import prepare_surfaces

    c = copy.deepcopy(cfg)
    c["intruder"]["outer_band_m"] = float(band)
    for sensor in c["sensors"]["cameras"] + c["sensors"]["radars"]:
        sensor["alpha"] = float(sensor["alpha"]) * float(foliage)
    s = prepare_surfaces(c)
    per = gpd.read_file(root / c["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    der = root / c["site"]["output_gpkg"]
    assets = gpd.read_file(der, layer="assets")
    access = gpd.read_file(der, layer="access")
    cand = gpd.read_file(root / c["candidates"]["output"])
    cells = np.load(root / c["coverage"]["out_dir"] / "cells.npz")
    dm = domain_cells(s, per, assets, cells, c)
    u, v, l, fr = build_edges(s.grid, dm["rows"], dm["cols"], dm["inside"])
    gate, _ = gate_edges(s.grid, dm["rows"], dm["cols"], u, v, fr, per, access)
    w = np.r_[cells["w"], np.zeros(dm["rows"].size - int(dm["inside"].sum()))]
    A, conf, *_ = build_matrix(s, cand, dm["rows"], dm["cols"], w, c)
    dom = Domain(rows=dm["rows"], cols=dm["cols"], inside=dm["inside"],
                 source=dm["source"], target=dm["target"], u=u, v=v, length=l,
                 crosses=fr, gate=gate)
    return dom, A.tocsr(), conf


def keys(conf: pd.DataFrame) -> np.ndarray:
    """Key (location, height, sensor, azimuth, tilt), the one of src.intruder_domain."""
    return config_key(conf).to_numpy()


def map_configs(T, keys_src: np.ndarray, index_dst: dict) -> tuple[np.ndarray, int]:
    """Indices in the scenario; the missing configurations (no detection) are counted."""
    out = [index_dst[k] for k in keys_src[np.asarray(T, np.int64)] if k in index_dst]
    return np.array(sorted(out), np.int64), len(T) - len(out)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


# =============================================================================
# Step 2: evaluation of a scenario
# =============================================================================
def evaluate_scenario(name, factor, value, cfg, dom, A, conf, ic_s, decisions, t_response,
                      theta, seed_keys, pool, max_time_s, verbose=False) -> list[dict]:
    """One row per T_R: certified c*_s, and regret of the nominal decision if it exists."""
    from src.site import ROOT
    cells = np.load(ROOT / cfg["coverage"]["out_dir"] / "cells.npz")
    n_int = int(dom.inside.sum())
    inst = from_configs(A[:, :n_int], cells["w"], conf, cfg)
    k_s = keys(conf)
    idx = {k: i for i, k in enumerate(k_s)}
    seeds = [g for g in (keys_to_indices(gk, idx) for gk in seed_keys) if g.size]
    d = time_remaining(dom, ic_s, PROFILE)
    T_min = float(d[dom.source].min())
    rows = []
    for T_R in t_response:
        base = dict(scenario=name, factor=factor, value=value, T_R=T_R, T_min=T_min)
        D = decisions.get(T_R)
        if D is not None:
            D_s = np.array(sorted(idx[k] for k in D["keys"] if k in idx), np.int64)
            p_D = p_tilde(dom, A, D_s, d, T_R, PROFILE) if D_s.size else 0.0
            base.update(cost_D=D["cost"], p_D=p_D, holds=bool(p_D >= theta - 1e-9),
                        n_missing=len(D["keys"]) - D_s.size)
        if T_R >= T_min:                       # proposition 2: impossible, no computation
            rows.append(dict(base, status="impossible (T_R ≥ T_min)", c_star=np.nan))
            continue
        t0 = time.perf_counter()
        r = adversarial_loop(inst, dom, A, d, theta, T_R, PROFILE, "cost",
                             seeds=seeds + ([D_s] if D is not None and D_s.size else []),
                             n_paths=20, max_time_s=max_time_s, master_time_s=300, fusion="max",
                             pool=pool, verbose=verbose)
        row = dict(base, status=r.status, c_star=r.value, bound=r.bound,
                   time_s=time.perf_counter() - t0, iterations=r.iterations,
                   D_star=" ".join(k_s[r.T]) if r.T.size else "")
        if r.T.size:
            chk = p_tilde(dom, A, r.T, d, T_R, PROFILE)
            assert r.status != "optimal" or chk >= theta - 1e-6, f"{name}: p̃* = {chk}"
            kinds = conf.iloc[r.T]["sensor"].str.startswith("radar")
            row.update(n_radar=int(kinds.sum()), n_camera=int((~kinds).sum()))
        if D is not None and r.T.size:
            row["jaccard"] = jaccard(set(D["keys"]), set(k_s[r.T]))
            if row["holds"]:
                row["regret"] = D["cost"] - r.value
            else:
                row["extra_cost"] = r.value - D["cost"]
        rows.append(row)
        print(f"  {name:<22} T_R = {T_R:>5g} s | T_min {T_min:5.1f} s | c* = {r.value:6.2f} "
              f"({r.status}, {row['time_s']:.0f} s)"
              + (f" | D: p̃* = {base['p_D']:.3f}, {'holds' if base['holds'] else 'DOES NOT HOLD'}"
                 if D is not None else ""), flush=True)
    return rows


def keys_to_indices(gk: np.ndarray, idx: dict) -> np.ndarray:
    return np.array(sorted(idx[k] for k in gk if k in idx), np.int64)


# =============================================================================
# Step 3: figures
# =============================================================================
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e2dd"
PAL = ("#2a78d6", "#1f9e6e", "#eb6834", "#b8456a", "#4a3aa7")
C_OK, C_KO = "#1f9e6e", "#d03b3b"


def plot_frontier(fr: pd.DataFrame, path):
    """Delay × response time matrix: certified minimal cost, or proven impossibility."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    lines = list(dict.fromkeys(fr["value"]))
    cols = sorted(fr["T_R"].unique())
    M = np.full((len(lines), len(cols)), np.nan)
    tmin = {}
    for _, r in fr.iterrows():
        i, j = lines.index(r["value"]), cols.index(r["T_R"])
        tmin[r["value"]] = r["T_min"]
        if r["status"] == "optimal":
            M[i, j] = r["c_star"]
    cmap = LinearSegmentedColormap.from_list("c", ["#dcebfa", "#2a78d6"])
    cmap.set_bad("#f1efea")
    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=150, layout="constrained")
    lo, hi = (np.nanmin(M), np.nanmax(M)) if np.isfinite(M).any() else (0.0, 1.0)
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=lo - 0.5, vmax=hi + 0.5, aspect="auto")
    for i in range(len(lines)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9, color=INK)
            else:
                ax.text(j, i, "impossible", ha="center", va="center", fontsize=7.5,
                        color="#8a8883", style="italic")
    ax.set_xticks(range(len(cols)), [f"{c:g} s" for c in cols])
    ax.set_yticks(range(len(lines)),
                  [f"{v} s  (T_min = {tmin[v]:.0f} s)" for v in lines])
    ax.set_xlabel("Response time T_R", color=INK_2)
    ax.set_ylabel("Delay (climb, gate)", color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Certified minimal cost for a guaranteed timely detection ≥ 0.9",
                 loc="left", fontsize=9, color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_regret(rg: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decs = sorted(rg["T_R"].unique())
    fig, axes = plt.subplots(1, len(decs), figsize=(4.4 * len(decs), 4.2), dpi=150,
                             sharey=True, layout="constrained")
    for ax, T_R in zip(np.atleast_1d(axes), decs):
        g = rg[rg["T_R"] == T_R].reset_index(drop=True)
        y = np.arange(len(g))[::-1]
        cD = g["cost_D"].iloc[0]
        ax.axvline(cD, color=INK_2, lw=1, ls="--")
        for yi, (_, r) in zip(y, g.iterrows()):
            col = C_OK if r["holds"] else C_KO
            if np.isfinite(r["c_star"]):
                ax.plot([r["c_star"], cD], [yi, yi], color=col, lw=1.5, alpha=0.5)
                ax.plot(r["c_star"], yi, "o", color=col, ms=6)
            lab = f"p̃* = {r['p_D']:.2f}" + ("" if r["holds"] else "  ✗")
            x_txt = max(cD, r["c_star"]) if np.isfinite(r["c_star"]) else cD
            ax.text(x_txt + 0.3, yi, lab, va="center", fontsize=7, color=col)
        ax.set_yticks(y); ax.set_yticklabels(g["scenario"], fontsize=8)
        ax.set_title(f"Decision D_{T_R:g} (cost {cD:g}, dashed)", loc="left", fontsize=9,
                     color=INK)
        ax.set_xlabel("Certified minimal cost under the scenario c*_s", color=INK_2)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xlim(right=ax.get_xlim()[1] + 3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Green: the nominal decision holds (gap = regret); red: it does not "
                 "hold (gap = extra cost needed)", fontsize=8, color=INK_2, x=0.01, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figures(root, cfg):
    out = root / cfg["sensitivity"]["out_dir"]
    fig_dir = root / cfg["optimization"]["figures_dir"]
    fr = pd.read_csv(out / "frontier.csv")
    plot_frontier(fr, fig_dir / "delay_frontier.png")
    try:
        rg = pd.read_csv(out / "regret.csv")
    except pd.errors.EmptyDataError:            # no nominal decision: nothing to plot
        return
    rg = rg[rg["T_R"].isin(cfg["sensitivity"]["decisions_t_response_s"])]
    if len(rg):
        plot_regret(rg, fig_dir / "regret.png")


# =============================================================================
# Step 4: main program
# =============================================================================
def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        figures(ROOT, cfg)
        return
    sc, ic = cfg["sensitivity"], cfg["intruder"]
    out = ROOT / sc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    theta = float(sc["theta"])

    # Nominal decisions and seeds, identified by key
    dom0, A0, conf0 = load_domain(cfg, ROOT)
    k0 = keys(conf0)
    sol = pd.read_csv(ROOT / cfg["loop"]["out_dir"] / "solutions.csv")
    cells = np.load(ROOT / cfg["coverage"]["out_dir"] / "cells.npz")
    inst0 = from_configs(A0[:, :int(dom0.inside.sum())], cells["w"], conf0, cfg)
    decisions = {}
    for T_R in sc["decisions_t_response_s"]:
        T = sol[(sol["fusion"] == "max") & np.isclose(sol["theta"], theta)
                & np.isclose(sol["T_R"], T_R)]["config"].to_numpy(np.int64)
        if not T.size:                  # infeasible (T_R ≥ T_min, e.g. cameras only) or not run
            print(f"D_{T_R:g}: no certified decision in {cfg['loop']['out_dir']}/solutions.csv "
                  f"(θ = {theta:g}, T_R = {T_R:g} s), skipped")
            continue
        decisions[float(T_R)] = dict(keys=list(k0[T]), cost=cost(inst0, T))
        print(f"D_{T_R:g}: cost {decisions[float(T_R)]['cost']:g}, "
              f"{', '.join(k0[T])}")
    seed_keys = [k0[T] for T in front_nsga(ROOT, cfg)]
    seed_keys += [np.array(D["keys"]) for D in decisions.values()]
    t_dec = sorted(decisions)
    rows, frontier = [], []

    # 1. Delay and speed: same domain, only d changes; one shared pool
    pool0: list = []
    for tc, tp in sc["delay"]:
        ic_s = dict(ic, t_fence_s=float(tc), t_gate_s=float(tp))
        name = f"delay ({tc:g}, {tp:g}) s"
        t_all = sorted(set(t_dec) | set(map(float, sc["delay_t_response_s"])))
        r = evaluate_scenario(name, "delay", f"({tc:g}, {tp:g})", cfg, dom0, A0, conf0, ic_s,
                              decisions, t_all, theta, seed_keys, pool0, sc["max_time_s"])
        frontier += r
        rows += [x for x in r if x["T_R"] in t_dec]
    for speed in sc["speed_m_s"]:
        ic_s = dict(ic, speed_m_s=float(speed))
        rows += evaluate_scenario(f"speed {speed:g} m/s", "speed", speed, cfg, dom0, A0, conf0,
                                  ic_s, decisions, t_dec, theta, seed_keys, pool0,
                                  sc["max_time_s"])
    pd.DataFrame(frontier).to_csv(out / "frontier.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "regret.csv", index=False)

    # 2. Band and foliage: domain and A_intruder rebuilt; one pool per scenario
    for factor, values in (("band", sc["outer_band_m"]), ("foliage", sc["foliage"])):
        for val in values:
            t0 = time.perf_counter()
            b = float(val) if factor == "band" else float(ic["outer_band_m"])
            f = float(val) if factor == "foliage" else 1.0
            dom, A, conf = build_scenario(cfg, ROOT, b, f)
            print(f"{factor} = {val:g}: {dom.n:,} nodes, {A.shape[0]:,} configurations, "
                  f"{A.nnz:,} non-zeros ({time.perf_counter() - t0:.0f} s)", flush=True)
            name = f"band {val:g} m" if factor == "band" else f"foliage × {val:g}"
            rows += evaluate_scenario(name, factor, val, cfg, dom, A, conf, ic, decisions, t_dec,
                                      theta, seed_keys, [], sc["max_time_s"])
            pd.DataFrame(rows).to_csv(out / "regret.csv", index=False)

    rg = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    cols = ["scenario", "T_R", "T_min", "cost_D", "p_D", "holds", "c_star", "status", "regret",
            "extra_cost", "jaccard", "n_radar", "n_camera"]
    print(rg[[c for c in cols if c in rg]].round(3).to_string(index=False))
    for T_R, g in rg.groupby("T_R"):
        D = decisions[T_R]
        stable = g["holds"].all() and (g["regret"].fillna(0) < sc["stable_threshold"] * D["cost"]).all()
        print(f"D_{T_R:g}: {'STABLE' if stable else 'NOT STABLE'} "
              f"(holds in {int(g['holds'].sum())}/{len(g)} scenarios)")
    figures(ROOT, cfg)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
