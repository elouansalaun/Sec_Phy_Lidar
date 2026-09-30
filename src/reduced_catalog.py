"""
src/reduced_catalog.py: reduced catalogs: does the selection matter when candidate positions
are scarce? (§8.9, step 6.3c)

With many positions, a small site is saturated: a catalog drawn at random can do as well as the
manual catalog and as B2 (6.3b). Hypothesis: the advantage of an informed selection only shows
when the number of positions is small. We compare, for K positions (same mast / roof ratio as
the manual catalog):
  B2_weighted, oracle_weighted : non-maximum suppression on the weighted score (src.surrogate_catalog)
  random × seeds               : possible positions drawn at random, same minimal spacing
  manual_sub × seeds           : K positions drawn from the week 1 manual catalog
Exact and identical evaluation: matrix on the intruder domain, greedy over the budget grid,
p*_F and p*_G of each solution (no NSGA-II).

Mode `protocol: "preregistered"` (6.3d, decided BEFORE running, config surrogate.decision.reduced):
  spacing      R_type = factor · √(A_type / K_type), A_type = area of the possible positions of
               the type (same R for the selected catalogs AND for the random control)
  band         exterior cells of the intruder domain added to the weight map, total mass
               = band_mass × interior mass, spread uniformly
  catalogs     B2_band (candidate), oracle_band, B2_weighted (no band), random, manual_sub
  success      mean F of B2_band > max of the random seeds for ≥ 3 of the 4 sizes
Mode `protocol: "initial"` (6.3c): fixed spacing `nms_m`, no band.

Outputs (surrogate.out_dir/decision): reduced.csv (one row per catalog and budget),
                                      reduced_summary.csv; figure: figures/decision_reduced.png
  (preregistered mode: suffix _preregistered on all three)
Usage : python -m src.reduced_catalog
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from src.coverage import build_matrix
from src.greedy import greedy
from src.horizons import Horizons
from src.instance import from_configs, violations
from src.intruder import corridor_threshold, detection
from src.site import ROOT, load_config
from src.surrogate_baselines import predict_b2
from src.surrogate_catalog import (CatalogContext, load_b2, manual_counts, possible_positions,
                                   selection, to_candidates, weighted_mass, weighted_scores)

COLORS = {"B2_weighted": "#2a78d6", "oracle_weighted": "#1baf7a", "random": "#eb6834",
          "manual_sub": "#4a3aa7", "B2_band": "#2a78d6", "oracle_band": "#1baf7a"}
LINESTYLES = {"B2_weighted": "--"}
LABELS = {"B2_weighted": "B2, weighted score", "oracle_weighted": "Oracle, weighted score",
          "random": "Random (5 seeds)", "manual_sub": "Manual sub-catalog (5 seeds)",
          "B2_band": "B2, weighted score + band", "oracle_band": "Oracle, weighted score + band"}
INK, INK_2 = "#0b0b0b", "#52514e"


# =============================================================================
# Step 0: preregistered protocol: weight map with band, spacing by rule
# =============================================================================
def weight_map(cfg, band_mass: float) -> np.ndarray:
    """Interior weights (cells.npz) + outer band of the intruder domain, of total mass
    band_mass × interior mass spread uniformly."""
    g = cfg["grid"]
    cel = np.load(ROOT / cfg["coverage"]["out_dir"] / "cells.npz")
    dm = np.load(ROOT / cfg["intruder"]["out_dir"] / "domain.npz")
    W = np.zeros((g["height"], g["width"]))
    W[cel["rows"], cel["cols"]] = cel["w"]
    ext = ~dm["inside"]
    if band_mass > 0:
        W[dm["rows"][ext], dm["cols"][ext]] = band_mass * cel["w"].sum() / ext.sum()
    return W


def mass(cfg, W, rows, cols, batch=100):
    """M (n, ring·sector): sum of W over each bin of the disc (cf. surrogate_catalog)."""
    from scipy import sparse

    from src.surrogate_dataset import disc_bins
    sc = cfg["surrogate"]
    dr, dc, b, n_bin = disc_bins(sc["radius_m"], sc["rings_m"], sc["n_sectors"],
                                 cfg["site"]["resolution_m"])
    onehot = sparse.csr_matrix((np.ones(b.size), (np.arange(b.size), b)),
                               shape=(b.size, n_bin.size))
    M = np.zeros((rows.size, n_bin.size))
    for a in range(0, rows.size, batch):
        M[a:a + batch] = W[rows[a:a + batch, None] + dr[None], cols[a:a + batch, None] + dc[None]] @ onehot
    return M


def radii(mast, roof, n_mast, n_roof, factor, step_m=2.0):
    """R = factor · √(A / K) per type; A = number of possible positions × step²."""
    return (factor * np.sqrt(mast.sum() * step_m ** 2 / max(n_mast, 1)),
            factor * np.sqrt(roof.sum() * step_m ** 2 / max(n_roof, 1)))


def selection2(score, x, y, mast, roof, n_mast, n_roof, R_mast, R_roof, rng=None):
    from src.surrogate_metrics import nms
    out = []
    for m, k, R in ((mast, n_mast, R_mast), (roof, n_roof, R_roof)):
        idx = np.flatnonzero(m)
        sc = rng.random(idx.size) if rng is not None else score[idx]
        j = nms(sc, x[idx], y[idx], k, R)
        assert j.size == k, f"{j.size} positions spaced by {R:.1f} m (wanted: {k})"
        out.append(idx[j])
    return out


# =============================================================================
# Step 1: exact greedy evaluation of a catalog
# =============================================================================
def evaluate_greedy(ctx: CatalogContext, cand) -> pd.DataFrame:
    cfg = ctx.cfg
    A, conf, _, _ = build_matrix(ctx.s, cand, ctx.dom.rows, ctx.dom.cols, ctx.w_dom, cfg)
    A = A.tocsr()
    inst = from_configs(A[:, :ctx.n_int], ctx.w, conf, cfg)
    rows = []
    for B in ctx.budgets:
        sol = greedy(inst, B)
        assert not violations(inst, sol.T, B)
        P, _ = detection(A, sol.T)
        rows.append(dict(budget=B, cost=sol.cost, F=sol.value / inst.W,
                         p_star_F=corridor_threshold(ctx.dom, P, "F")[0] if sol.T.size else 0.0,
                         p_star_G=corridor_threshold(ctx.dom, P, "G")[0] if sol.T.size else 0.0))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (K, catalog, seed), then mean / min / max over the seeds."""
    def one(g):
        rob = g[g.p_star_F >= 0.9]
        return pd.Series(dict(F_mean=g.F.mean(),
                              F_B13=g.loc[g.budget == 13, "F"].iloc[0],
                              F_B20=g.loc[g.budget == 20, "F"].iloc[0],
                              pF_B13=g.loc[g.budget == 13, "p_star_F"].iloc[0],
                              pF_B20=g.loc[g.budget == 20, "p_star_F"].iloc[0],
                              robust_budget=rob.budget.min() if len(rob) else np.nan))
    per = df.groupby(["K", "catalog", "seed"]).apply(one, include_groups=False).reset_index()
    agg = per.groupby(["K", "catalog"]).agg(["mean", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return per, agg.drop(columns=[c for c in agg.columns if c.startswith("seed")]).reset_index()


# =============================================================================
# Step 2: figure
# =============================================================================
def figure(per: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.intruder_evaluation import style

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    for ax, col, title, pct in zip(axs, ("F_mean", "F_B13", "robust_budget"),
                                   ("Mean F over the budgets (greedy)", "F at B = 13",
                                    "First budget with p*_F ≥ 0.9"), (True, True, False)):
        style(ax)
        Ks = sorted(per.K.unique())
        for j, (name, g) in enumerate(per.groupby("catalog", sort=False)):
            n_c = per.catalog.nunique()
            dx = (j - (n_c - 1) / 2) * 0.6 / n_c
            m = g.groupby("K")[col].agg(["mean", "min", "max"]).reindex(Ks)
            k = 100 if pct else 1
            xs = np.arange(len(Ks)) + dx
            ax.plot(xs, k * m["mean"], color=COLORS[name], lw=2, marker="o", ms=5,
                    ls=LINESTYLES.get(name, "-"), label=LABELS[name])
            if (m["max"] > m["min"]).any():
                ax.vlines(xs, k * m["min"], k * m["max"], color=COLORS[name], lw=1.2, alpha=0.8)
        ax.set_xticks(range(len(Ks)), [f"K = {k}" for k in Ks])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        if pct:
            ax.set_ylabel("%")
    axs[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.suptitle("Reduced catalogs: mean and range over the seeds (vertical bars)",
                 x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Step 3: orchestration
# =============================================================================
def main() -> None:
    cfg = load_config()
    cfg["grid"] = yaml.safe_load(open(ROOT / cfg["site"]["grid_yaml"]))
    sc, dc = cfg["surrogate"], cfg["surrogate"]["decision"]
    rc = dc["reduced"]
    out = ROOT / sc["out_dir"] / "decision"
    out.mkdir(parents=True, exist_ok=True)

    te = np.load(ROOT / sc["out_dir"] / "test.npz")
    x, y, mast, roof, building_name = possible_positions(cfg, te)
    n_mast61, n_roof61, manual = manual_counts(cfg)
    ctx = CatalogContext(cfg)
    pos = np.flatnonzero(mast | roof)
    V2 = np.zeros_like(te["Y"])
    V2[pos] = predict_b2(load_b2(ROOT / sc["out_dir"] / "b2.joblib"), Horizons(ctx.s, cfg),
                         te["rows"][pos], te["cols"][pos], te["stratum"][pos].astype(int),
                         te["heights"])
    M = np.zeros((te["rows"].size, V2[0].size // len(te["heights"])))
    M[pos] = weighted_mass(cfg, te["rows"][pos].astype(np.int64), te["cols"][pos].astype(np.int64))
    w_b2 = weighted_scores(V2, te["heights"], cfg, mast, roof, M)
    w_ex = weighted_scores(te["Y"], te["heights"], cfg, mast, roof, M)
    ok = np.isfinite(w_ex)
    print(f"Weighted score: Spearman B2 / exact = {spearmanr(w_b2[ok], w_ex[ok])[0]:.3f}")

    m_man = (manual["support"] != "roof").to_numpy()
    prereg = rc.get("protocol", "initial") == "preregistered"
    if prereg:
        W = weight_map(cfg, rc["band_mass"])
        Mb = np.zeros_like(M)
        Mb[pos] = mass(cfg, W, te["rows"][pos].astype(np.int64), te["cols"][pos].astype(np.int64))
        b_b2 = weighted_scores(V2, te["heights"], cfg, mast, roof, Mb)
        b_ex = weighted_scores(te["Y"], te["heights"], cfg, mast, roof, Mb)
        print(f"Weighted score + band: Spearman B2 / exact = {spearmanr(b_b2[ok], b_ex[ok])[0]:.3f}; "
              f"with / without band (exact) = {spearmanr(b_ex[ok], w_ex[ok])[0]:.3f}")
        chosen = (("B2_band", b_b2), ("oracle_band", b_ex), ("B2_weighted", w_b2))
    else:
        chosen = (("B2_weighted", w_b2), ("oracle_weighted", w_ex))

    rows, t0 = [], time.time()
    for K in (rc["preregistered_sizes"] if prereg else rc["sizes"]):
        n_mast = int(round(K * n_mast61 / (n_mast61 + n_roof61)))
        n_roof = K - n_mast
        if prereg:
            Rm, Rr = radii(mast, roof, n_mast, n_roof, rc["spacing_factor"])
        else:
            Rm = Rr = dc["nms_m"]
        print(f"K = {K}: {n_mast} masts (spacing {Rm:.1f} m), {n_roof} roofs ({Rr:.1f} m)")
        cats = []
        for name, sc_ in chosen:
            i_m, i_r = selection2(sc_, x, y, mast, roof, n_mast, n_roof, Rm, Rr)
            cats.append((name, 0, to_candidates(cfg, x, y, i_m, i_r, building_name, "R")))
        for g in rc["seeds"]:
            rng = np.random.default_rng(g)
            i_m, i_r = selection2(None, x, y, mast, roof, n_mast, n_roof, Rm, Rr, rng)
            cats.append(("random", g, to_candidates(cfg, x, y, i_m, i_r, building_name, "R")))
            idx = np.r_[rng.choice(np.flatnonzero(m_man), n_mast, replace=False),
                        rng.choice(np.flatnonzero(~m_man), n_roof, replace=False)]
            cats.append(("manual_sub", g, manual.iloc[idx].reset_index(drop=True)))
        for name, g, c in cats:
            df = evaluate_greedy(ctx, c).assign(K=K, catalog=name, seed=g)
            rows.append(df)
            print(f"  K = {K:2d}  {name:15s} seed {g}  mean F {df.F.mean():.3f}   "
                  f"p*_F(B=20) {df.loc[df.budget == 20, 'p_star_F'].iloc[0]:.2f}"
                  f"   [{time.time() - t0:5.0f} s]", flush=True)
    df = pd.concat(rows, ignore_index=True)
    per, agg = summarize(df)
    if not prereg:                                  # 6.3d writes its own files (below)
        df.to_csv(out / "reduced.csv", index=False)
        agg.to_csv(out / "reduced_summary.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["K", "catalog"] + [c for c in agg.columns if c.endswith("_mean")]
    print("\n" + agg[cols].round(3).to_string(index=False))
    suffix = "_preregistered" if prereg else ""
    if prereg:
        # Preregistered success criterion
        ok_K = []
        for K, g in per.groupby("K"):
            b2 = g.loc[g.catalog == "B2_band", "F_mean"].iloc[0]
            rnd = g.loc[g.catalog == "random", "F_mean"]
            ok_K.append(b2 > rnd.max())
            print(f"  K = {K:2d}: B2_band {b2:.3f}  vs random max {rnd.max():.3f} "
                  f"(mean {rnd.mean():.3f}) -> {'beats' if b2 > rnd.max() else 'does not beat'}")
        verdict = sum(ok_K) >= 3
        print(f"PREREGISTERED CRITERION (≥ 3 sizes out of 4): {sum(ok_K)} / {len(ok_K)} -> "
              f"{'SUCCESS' if verdict else 'FAILURE'}")
        agg.to_csv(out / "reduced_summary_preregistered.csv", index=False)
        df.to_csv(out / "reduced_preregistered.csv", index=False)
    figure(per, ROOT / cfg["optimization"]["figures_dir"] / f"decision_reduced{suffix}.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
