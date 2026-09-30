"""
src/surrogate_catalog.py: validation by the decision (§8.9, step 6.3).

Question: does a catalog of candidate positions chosen by the surrogate model B2 give better
layouts than the catalog made by hand in week 1?

Six catalogs, with the SAME counts as the week 1 catalog (masts and roofs):
  manual          : data/site/candidates.gpkg (week 1)
  random          : possible positions drawn at random (what the selection itself is worth)
  B2, oracle      : non-maximum suppression on the GEOMETRIC potential V̄, predicted by B2 or
                    exact (6.3). Result: worse than the manual one, because V̄ ignores the weighted zones.
  B2_weighted, oracle_weighted : same on the WEIGHTED score (6.3b): each bin of V is weighted by
                    the mass of weight w it contains, computed analytically; the surrogate model
                    is unchanged, only the knowledge of the site is added.

Possible positions (week 1 rules), on the grid of the test block (step of 2 m):
  masts : ground inside the perimeter, ≥ `inset_m` from the fence and ≥ `min_dist_building_m`
          from the buildings; heights 4 to 10 m
  roofs : roof cells ≥ `inset_m` (roof) from the building edge; heights 2 and 4 m
Geometric score of a position: max, over its possible heights, of the total potential V̄_h.
Weighted score: with M_b(o) = Σ_{j ∈ bin b} w_j,
  radar   R_h(o) = Σ_b M_b(o) · V_{h,b}(o)            (360°)
  camera  C_h(o) = max_s Σ_a M_{a,s}(o) · V_{h,a,s}(o) (best sector)
  score(o) = max_h max(R_h / max R, C_h / max C)       (h among the heights of the support;
                                                       C_h only with sensors.cameras_only)
Assumption: within a bin, visibility and weight are independent (V_b is a mean over the bin).

For each catalog, everything else is EXACT and identical: matrix on the intruder domain
(coverage.build_matrix, as in src.intruder_domain), greedy over the budget grid, p*_F and p*_G
of each solution, then NSGA-II seeded with the greedy solutions of the same catalog.

Outputs (surrogate.out_dir/decision): candidates_<name>.gpkg, greedy_<name>.csv, front_<name>.csv,
                                      summary.csv; figures: decision_budget.png, decision_front.png,
                                      decision_map.png
Usage : python -m src.surrogate_catalog                              (6 catalogs)
        python -m src.surrogate_catalog B2_weighted oracle_weighted  (subset; the others are
        read back from decision/ if they have already been evaluated)
"""
from __future__ import annotations

import sys
import time

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
from scipy import sparse

from src.candidates import sample
from src.coverage import build_matrix
from src.greedy import greedy
from src.horizons import Horizons
from src.instance import KINDS, from_configs, violations
from src.intruder import Domain, corridor_threshold, detection
from src.site import ROOT, load_config
from src.surrogate_baselines import predict_b2
from src.surrogate_metrics import bin_areas, nms, total_potential
from src.viewshed import prepare_surfaces

CATALOGS = ("manual", "random", "B2", "oracle", "B2_weighted", "oracle_weighted")
COLORS = {"manual": "#4a3aa7", "random": "#eb6834", "B2": "#2a78d6", "oracle": "#1baf7a",
          "B2_weighted": "#2a78d6", "oracle_weighted": "#1baf7a"}
LINESTYLES = {"B2": "--", "oracle": "--"}               # geometric score: dashed
MARKERS = {"manual": "o", "random": "D", "B2": "s", "oracle": "^", "B2_weighted": "s",
           "oracle_weighted": "^"}
LABELS = {"manual": "Manual (week 1)", "random": "Random",
          "B2": "B2, geometric score", "oracle": "Oracle, geometric score",
          "B2_weighted": "B2, weighted score", "oracle_weighted": "Oracle, weighted score"}
INK, INK_2 = "#0b0b0b", "#52514e"


# =============================================================================
# Step 1: possible positions and scores
# =============================================================================
def possible_positions(cfg, te):
    """Positions of the test block (step of 2 m) where a mast or a roof support can be placed."""
    g = cfg["grid"]
    x = g["xmin"] + (te["cols"] + 0.5) * g["res"]
    y = g["ymax"] - (te["rows"] + 0.5) * g["res"]
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="assets")
    bld = assets[assets["type"] == "building"]
    cm, cr = cfg["candidates"]["mast"], cfg["candidates"]["roof"]
    roof_raster = sample(ROOT / cfg["rasters"]["out_dir"] / "building_mask.tif", np.c_[x, y]) == 1

    mast = (shapely.contains_xy(per.buffer(-cm["inset_m"]), x, y)
            & ~shapely.contains_xy(bld.union_all().buffer(cm["min_dist_building_m"]), x, y)
            & (te["stratum"] != 0))
    building_name = np.full(x.size, None, object)
    for _, b in bld.iterrows():
        m = shapely.contains_xy(b.geometry.buffer(-cr["inset_m"]), x, y) & roof_raster
        building_name[m] = b["name"]
    roof = (building_name != None) & (te["stratum"] == 0)                  # noqa: E711
    return x, y, mast, roof, building_name


def scores(V, heights, cfg, mast, roof):
    """max over the possible heights of the support of V̄_h (V: (n, h, ring, sector))."""
    Vb = total_potential(V, bin_areas(cfg))                              # (n, h)
    h = np.asarray(heights, float)
    hm = np.isin(h, cfg["candidates"]["mast"]["heights_m"])
    hr = np.isin(h, cfg["candidates"]["roof"]["heights_m"])
    return np.where(mast, Vb[:, hm].max(1), np.where(roof, Vb[:, hr].max(1), -np.inf))


def weighted_mass(cfg, rows, cols, batch=100):
    """M (n, ring·sector): sum of the weights w of the cells of each bin of the disc of o."""
    from src.surrogate_dataset import disc_bins
    sc = cfg["surrogate"]
    dr, dc, b, n_bin = disc_bins(sc["radius_m"], sc["rings_m"], sc["n_sectors"],
                                 cfg["site"]["resolution_m"])
    cel = np.load(ROOT / cfg["coverage"]["out_dir"] / "cells.npz")
    g = cfg["grid"]
    Wmap = np.zeros((g["height"], g["width"]))
    Wmap[cel["rows"], cel["cols"]] = cel["w"]
    onehot = sparse.csr_matrix((np.ones(b.size), (np.arange(b.size), b)), shape=(b.size, n_bin.size))
    M = np.zeros((rows.size, n_bin.size))
    for a in range(0, rows.size, batch):
        G = Wmap[rows[a:a + batch, None] + dr[None], cols[a:a + batch, None] + dc[None]]
        M[a:a + batch] = G @ onehot                      # dense × sparse -> dense
    return M


def weighted_scores(V, heights, cfg, mast, roof, M):
    """Weighted score (module docstring). V: (n, h, ring, sector), M: (n, ring·sector)."""
    n, nh, na, ns = V.shape
    Mv = V * M.reshape(n, 1, na, ns)
    R = Mv.sum(axis=(2, 3))                                  # (n, h)
    C = Mv.sum(axis=2).max(axis=2)                           # (n, h)
    ok = mast | roof
    S = C / C[ok].max()
    if cfg["sensors"]["radars"]:                             # otherwise cameras only
        S = np.maximum(R / R[ok].max(), S)
    h = np.asarray(heights, float)
    hm = np.isin(h, cfg["candidates"]["mast"]["heights_m"])
    hr = np.isin(h, cfg["candidates"]["roof"]["heights_m"])
    return np.where(mast, S[:, hm].max(1), np.where(roof, S[:, hr].max(1), -np.inf))


# =============================================================================
# Step 2: catalogs
# =============================================================================
def manual_counts(cfg):
    c = gpd.read_file(ROOT / cfg["candidates"]["output"])
    return int((c["support"] != "roof").sum()), int((c["support"] == "roof").sum()), c


def selection(score, x, y, mast, roof, n_mast, n_roof, radius, rng=None):
    """Selected indices: non-maximum suppression per type (or random draw if rng)."""
    out = []
    for m, k in ((mast, n_mast), (roof, n_roof)):
        idx = np.flatnonzero(m)
        if rng is not None:
            sc = rng.random(idx.size)                    # random order, same minimal spacing
        else:
            sc = score[idx]
        sel = nms(sc, x[idx], y[idx], k, radius)
        assert sel.size == k, f"only {sel.size} positions spaced by {radius} m (wanted: {k})"
        out.append(idx[sel])
    return out


def to_candidates(cfg, x, y, i_mast, i_roof, building_name, prefix):
    """Same schema as data/site/candidates.gpkg (src.candidates.build_candidates)."""
    cm, cr = cfg["candidates"]["mast"], cfg["candidates"]["roof"]
    d = ROOT / cfg["rasters"]["out_dir"]
    idx = np.r_[i_mast, i_roof]
    is_roof = np.r_[np.zeros(i_mast.size, bool), np.ones(i_roof.size, bool)]
    xy = np.c_[x[idx], y[idx]]
    gdf = gpd.GeoDataFrame({"support": np.where(is_roof, "roof", "interior_mast"),
                            "building": np.where(is_roof, building_name[idx], None)},
                           geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=cfg["lidar"]["crs"])
    gdf["z_ground"] = sample(d / "dtm.tif", xy)
    gdf["z_dsm"] = sample(d / "dsm.tif", xy)
    gdf["h_vegetation"] = np.where(is_roof, np.nan, sample(d / "other_h.tif", xy))
    gdf["z_support"] = np.where(is_roof, gdf["z_dsm"], gdf["z_ground"])
    gdf["heights_m"] = np.where(is_roof, ",".join(map(str, cr["heights_m"])),
                                ",".join(map(str, cm["heights_m"])))
    gdf["cost_base"] = np.where(is_roof, cr["cost_base"], cm["cost_base"])
    gdf["cost_per_m"] = np.where(is_roof, cr["cost_per_m"], cm["cost_per_m"])
    gdf.insert(0, "id", [f"{prefix}{i:03d}" for i in range(len(gdf))])
    return gdf


# =============================================================================
# Step 3: exact evaluation of a catalog: matrix, greedy, worst path, NSGA-II
# =============================================================================
class CatalogContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.s = prepare_surfaces(cfg)
        d = ROOT / cfg["intruder"]["out_dir"]
        dm, gr = np.load(d / "domain.npz"), np.load(d / "graph.npz")
        self.dom = Domain(rows=dm["rows"], cols=dm["cols"], inside=dm["inside"],
                          source=dm["source"], target=dm["target"], u=gr["u"], v=gr["v"],
                          length=gr["length"], crosses=gr["crosses"], gate=gr["gate"])
        self.w_dom = dm["w"]
        self.n_int = int(self.dom.inside.sum())
        assert self.dom.inside[:self.n_int].all()
        cells = np.load(ROOT / cfg["coverage"]["out_dir"] / "cells.npz")
        self.w, self.zone = cells["w"], cells["zone"]
        b = cfg["optimization"]["budget"]
        self.budgets = np.arange(b["min"], b["max"] + 1e-9, b["step"])


def evaluate_catalog(ctx: CatalogContext, cand, name: str, out) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = ctx.cfg
    t0 = time.time()
    A, conf, _, _ = build_matrix(ctx.s, cand, ctx.dom.rows, ctx.dom.cols, ctx.w_dom, cfg)
    A = A.tocsr()
    t_A = time.time() - t0
    inst = from_configs(A[:, :ctx.n_int], ctx.w, conf, cfg)
    print(f"  {name}: {len(cand)} positions, {A.shape[0]:,} configurations ({t_A:.0f} s)")

    rows, seeds = [], []
    for B in ctx.budgets:
        sol = greedy(inst, B)
        assert not violations(inst, sol.T, B)
        P, _ = detection(A, sol.T)
        k = inst.kind[sol.T]
        rows.append(dict(catalog=name, budget=B, cost=sol.cost, F=sol.value / inst.W,
                         p_star_F=corridor_threshold(ctx.dom, P, "F")[0] if sol.T.size else 0.0,
                         p_star_G=corridor_threshold(ctx.dom, P, "G")[0] if sol.T.size else 0.0,
                         n_radar=int((k == KINDS.index("radar")).sum()),
                         n_camera=int((k == KINDS.index("camera")).sum()),
                         configs=" ".join(map(str, sol.T))))
        seeds.append(sol.T)
    gr = pd.DataFrame(rows)
    front = nsga(ctx, A, conf, inst, seeds, name)
    return gr, front


def nsga(ctx: CatalogContext, A, conf, inst, seeds, name) -> pd.DataFrame:
    """NSGA-II of src.nsga, on this catalog, seeded with its own greedy solutions."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    from src.nsga import (Evaluator, NeighbourhoodMutation, Placement, RepairPlacement,
                          SeededSampling, UnionCrossover)

    cfg, dc = ctx.cfg, ctx.cfg["surrogate"]["decision"]
    ev = Evaluator.__new__(Evaluator)                   # same evaluator, other catalog
    ev.cfg, ev.dom, ev.A, ev.conf = cfg, ctx.dom, A, conf
    ev.n_int, ev.inst, ev.zone = ctx.n_int, inst, ctx.zone
    k = cfg["nsga"]["k_coverage"]
    ev.k, ev.k_p, ev.k_share, ev.k_zones = k["k"], k["p_min"], k["min_share"], [int(z) for z in k["zones"]]
    ev.B_max = float(ctx.budgets[-1])

    fronts = []
    for g in dc["seeds"]:
        rng = np.random.default_rng(g)
        algo = NSGA2(pop_size=int(dc["population"]), sampling=SeededSampling(seeds, rng),
                     crossover=UnionCrossover(rng, cfg["nsga"]["p_crossover"]),
                     mutation=NeighbourhoodMutation(ev, rng), repair=RepairPlacement(ev, rng),
                     eliminate_duplicates=True)
        t0 = time.time()
        res = minimize(Placement(ev), algo, ("n_gen", int(dc["generations"])), seed=g,
                       verbose=False)
        for x in np.atleast_2d(res.X):
            T = np.flatnonzero(x.astype(bool))
            r = ev.evaluate(T)
            P, _ = detection(A, T)
            r["p_star_G"] = corridor_threshold(ctx.dom, P, "G")[0]
            fronts.append(dict(catalog=name, seed=g, **r, configs=" ".join(map(str, T))))
        print(f"    NSGA-II seed {g}: {time.time() - t0:.0f} s")
    f = pd.DataFrame(fronts).drop_duplicates("configs")
    from src.intruder_evaluation import non_dominated
    return f[non_dominated(f)].sort_values("cost").reset_index(drop=True)


# =============================================================================
# Step 4: summary and figures
# =============================================================================
def hypervolume(front, B_max):
    from pymoo.indicators.hv import HV
    Fo = np.c_[front.cost / B_max, -front.F, -front.p_star_F]
    return float(HV(ref_point=np.array([1.05, 0.0, 0.0]))(Fo))


def summary(gr: pd.DataFrame, fr: pd.DataFrame, B_max) -> pd.DataFrame:
    rows = []
    for name, g in gr.groupby("catalog", sort=False):
        f = fr[fr.catalog == name]
        r = dict(catalog=name, F_mean_greedy=g.F.mean())
        for B in (13, 20, 30):
            gg = g[g.budget == B].iloc[0]
            r[f"F_greedy_B{B}"], r[f"pF_greedy_B{B}"] = gg.F, gg.p_star_F
        rob = f[f.p_star_F >= 0.9]
        r["robustness_threshold_p0.9"] = rob.cost.min() if len(rob) else np.nan
        for B in (13, 20, 30):
            d = f[(f.cost <= B + 1e-9) & (f.p_star_F >= 0.95)]
            r[f"F_front_B{B}_p0.95"] = d.F.max() if len(d) else np.nan
        r["hypervolume"] = hypervolume(f, B_max)
        r["n_front"] = len(f)
        rows.append(r)
    return pd.DataFrame(rows)


def figures(gr, fr, cats, cfg, out_fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.intruder_evaluation import style

    lab = LABELS
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    for ax, col, title in zip(axs, ("F", "p_star_F"),
                              ("Weighted coverage F (greedy)", "Guaranteed detection p*_F (greedy)")):
        style(ax)
        for name, g in gr.groupby("catalog", sort=False):
            ax.plot(g.budget, 100 * g[col], color=COLORS[name], lw=2, marker=MARKERS[name],
                    ms=4, ls=LINESTYLES.get(name, "-"), label=lab[name],
                    mfc="white" if name in LINESTYLES else COLORS[name])
        ax.set_xlabel("Budget (cost units)")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axs[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_fig / "decision_budget.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    style(ax)
    for name, f in fr.groupby("catalog", sort=False):
        r = f[f.p_star_F >= 0.95].sort_values("cost")
        best = r.F.cummax()
        ax.step(r.cost, 100 * best, where="post", color=COLORS[name], lw=2,
                ls=LINESTYLES.get(name, "-"), label=lab[name])
        ax.scatter(r.cost, 100 * r.F, color=COLORS[name], marker=MARKERS[name], s=14, zorder=3)
    ax.set_xlabel("Cost (units)")
    ax.set_ylabel("Best coverage F (%)")
    ax.set_ylim(0, 100)
    ax.set_title("NSGA-II front restricted to robust layouts (p*_F ≥ 0.95)", loc="left",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_fig / "decision_front.png")
    plt.close(fig)

    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="assets")
    nc = 3
    nr = int(np.ceil(len(cats) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr), dpi=150, squeeze=False)
    for ax in axs.ravel()[len(cats):]:
        ax.axis("off")
    for ax, (name, (c, bg)) in zip(axs.ravel(), cats.items()):
        if bg is not None:
            x, y, v = bg
            ax.scatter(x, y, c=100 * v, s=2.2, marker="s", cmap="Blues", vmin=0,
                       vmax=np.nanpercentile(100 * v, 99), lw=0)
        assets[assets["type"] == "building"].plot(ax=ax, color="#d9d8d3", lw=0)
        ax.plot(*per.exterior.xy, color=INK, lw=1)
        for sup, mk in (("roof", "^"), ("mast", "o")):
            m = c["support"].str.contains(sup).to_numpy()
            ax.scatter(c.geometry.x[m], c.geometry.y[m], marker=mk, s=18, color=COLORS[name],
                       edgecolor="white", lw=0.6, zorder=3)
        ax.set_title(lab[name], loc="left", fontsize=10, color=INK)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Catalogs (circles: masts, triangles: roofs); background: score used for the "
                 "selection", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(out_fig / "decision_map.png")
    plt.close(fig)


# =============================================================================
# Step 5: orchestration
# =============================================================================
def load_b2(path):
    """The B2 model is a scikit-learn pickle: it only reloads with the version that wrote it.
    On failure, we ask to regenerate it in the current environment."""
    import warnings

    import sklearn
    import sklearn.exceptions
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", sklearn.exceptions.InconsistentVersionWarning)
            return joblib.load(path)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError,
            sklearn.exceptions.InconsistentVersionWarning) as e:
        raise SystemExit(f"{path.name} missing or written by another scikit-learn version "
                         f"({type(e).__name__}): first run `python -m src.surrogate_baselines` "
                         f"(scikit-learn {sklearn.__version__}).") from None


def main() -> None:
    import yaml

    cfg = load_config()
    cfg["grid"] = yaml.safe_load(open(ROOT / cfg["site"]["grid_yaml"]))
    sc, dc = cfg["surrogate"], cfg["surrogate"]["decision"]
    names = sys.argv[1:] or list(CATALOGS)
    out = ROOT / sc["out_dir"] / "decision"
    out.mkdir(parents=True, exist_ok=True)
    out_fig = ROOT / cfg["optimization"]["figures_dir"]
    out_fig.mkdir(parents=True, exist_ok=True)

    te = np.load(ROOT / sc["out_dir"] / "test.npz")
    x, y, mast, roof, building_name = possible_positions(cfg, te)
    n_mast, n_roof, manual = manual_counts(cfg)
    print(f"Possible positions (step of 2 m): {mast.sum():,} masts, {roof.sum():,} roofs; "
          f"counts of the manual catalog: {n_mast} masts, {n_roof} roofs")

    ctx = CatalogContext(cfg)
    hz = Horizons(ctx.s, cfg)
    pos = np.flatnonzero(mast | roof)
    t0 = time.time()
    V2 = np.zeros_like(te["Y"])
    V2[pos] = predict_b2(load_b2(ROOT / sc["out_dir"] / "b2.joblib"), hz, te["rows"][pos],
                         te["cols"][pos], te["stratum"][pos].astype(int), te["heights"])
    print(f"B2 map on {pos.size:,} positions: {time.time() - t0:.1f} s")
    s_b2 = scores(V2, te["heights"], cfg, mast, roof)
    s_ex = scores(te["Y"], te["heights"], cfg, mast, roof)
    ok = np.isfinite(s_ex)
    from scipy.stats import spearmanr
    print(f"B2 / exact agreement on the possible positions: Spearman {spearmanr(s_b2[ok], s_ex[ok])[0]:.3f}")

    M = np.zeros((te["rows"].size, V2[0].size // len(te["heights"])))
    M[pos] = weighted_mass(cfg, te["rows"][pos].astype(np.int64), te["cols"][pos].astype(np.int64))
    w_b2 = weighted_scores(V2, te["heights"], cfg, mast, roof, M)
    w_ex = weighted_scores(te["Y"], te["heights"], cfg, mast, roof, M)
    print(f"Weighted score: B2 / exact agreement Spearman {spearmanr(w_b2[ok], w_ex[ok])[0]:.3f}; "
          f"weighted / geometric agreement (exact) {spearmanr(w_ex[ok], s_ex[ok])[0]:.3f}")

    cat_scores = {"B2": s_b2, "oracle": s_ex, "B2_weighted": w_b2, "oracle_weighted": w_ex,
                  "random": None}
    cats = {}
    for name in CATALOGS:
        if name not in names:
            f = out / f"candidates_{name}.gpkg"
            if name == "manual":
                cats[name] = (manual, None)
            elif f.exists() and (out / f"greedy_{name}.csv").exists():
                cats[name] = (gpd.read_file(f), None)          # evaluated during a past run
            continue
        if name == "manual":
            cats[name] = (manual, None)
            continue
        rng = np.random.default_rng(dc["random_seed"]) if name == "random" else None
        i_m, i_r = selection(cat_scores[name], x, y, mast, roof, n_mast, n_roof, dc["nms_m"], rng)
        c = to_candidates(cfg, x, y, i_m, i_r, building_name, prefix=name[0].upper())
        c.to_file(out / f"candidates_{name}.gpkg", layer="candidates")
        sel = np.r_[i_m, i_r]
        print(f"  {name:15s}: mean exact V̄ {s_ex[sel].mean():.3f}; mean exact weighted score "
              f"{w_ex[sel].mean():.3f}")
        bg = cat_scores[name] if cat_scores[name] is not None else w_ex
        cats[name] = (c, (x[pos], y[pos], bg[pos]))

    grs, frs = [], []
    for name, (c, bg) in cats.items():
        if name in names:
            gr, fr = evaluate_catalog(ctx, c, name, out)
            gr.to_csv(out / f"greedy_{name}.csv", index=False)
            fr.to_csv(out / f"front_{name}.csv", index=False)
        else:
            gr = pd.read_csv(out / f"greedy_{name}.csv")
            fr = pd.read_csv(out / f"front_{name}.csv")
        grs.append(gr)
        frs.append(fr)
    gr, fr = pd.concat(grs), pd.concat(frs)
    r = summary(gr, fr, float(ctx.budgets[-1]))
    r.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + r.round(3).to_string(index=False))
    figures(gr, fr, cats, cfg, out_fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
