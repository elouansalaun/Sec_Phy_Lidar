"""
src/nsga_analysis.py: validation and reading of the NSGA-II front.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.intruder_evaluation import non_dominated, style

C_OPT, C_GREEDY = "#2a78d6", "#eb6834"
C_SEEDS = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7", "#d4a017"]
INK, INK_2 = "#0b0b0b", "#52514e"
THRESHOLDS = (0.9, 0.95)


# =============================================================================
# Step 1: global front and validation against the certified bound
# =============================================================================
def front_global(fronts: pd.DataFrame) -> pd.DataFrame:
    f = fronts.drop_duplicates("configs").reset_index(drop=True)
    return f[non_dominated(f)].sort_values("cost").reset_index(drop=True)


def bound(pareto: pd.DataFrame, c: np.ndarray, col: str) -> np.ndarray:
    """Value of `col` at the smallest budget of the grid ≥ c (UB increasing in B: valid bound)."""
    b = pareto.sort_values("budget")
    i = np.searchsorted(b.budget.to_numpy(), c - 1e-9)
    ok = i < len(b)
    out = np.full(c.size, np.nan)
    out[ok] = b[col].to_numpy()[i[ok]]
    return out


def validate(front: pd.DataFrame, pareto: pd.DataFrame) -> pd.DataFrame:
    front = front.copy()
    front["ub"] = bound(pareto, front.cost.to_numpy(), "ub")
    viol = front.F > front.ub + 1e-9
    assert not viol.any(), f"{viol.sum()} points above the certified bound: bug"
    return front


# =============================================================================
# Step 2: price of robustness
# =============================================================================
def price(sols: pd.DataFrame, budgets) -> pd.DataFrame:
    rows = []
    for B in budgets:
        d = sols[sols.cost <= B + 1e-9]
        r = dict(budget=B, F_max=d.F.max() if len(d) else 0.0)
        for s in THRESHOLDS:
            dd = d[d.p_star_F >= s]
            r[f"F_max_p{s:g}"] = dd.F.max() if len(dd) else np.nan
        # is the solution with the best F robust?
        r["p_star_of_F_max"] = d.loc[d.F.idxmax(), "p_star_F"] if len(d) else 0.0
        rows.append(r)
    p = pd.DataFrame(rows)
    for s in THRESHOLDS:
        p[f"price_p{s:g}_pt"] = 100 * (p.F_max - p[f"F_max_p{s:g}"])
    return p


# =============================================================================
# Step 3: figures
# =============================================================================
def plot_front(front, w3, pareto, path):
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    style(ax)
    b = pareto.sort_values("budget")
    ax.fill_between(b.budget, 100 * b.lb, 100 * b.ub, step="post", color=C_OPT, alpha=0.15, lw=0,
                    label="Week 3: certified optimum of F ∈ [LB, UB]")
    sc = ax.scatter(front.cost, 100 * front.F, c=100 * front.p_star_F, cmap="Blues", vmin=0,
                    vmax=100, s=38, edgecolor=INK_2, lw=0.5, zorder=3, label="NSGA-II front")
    m = w3[w3.method == "milp"]
    ax.scatter(m.cost, 100 * m.F, marker="x", s=26, color=C_GREEDY, lw=1.4, zorder=4,
               label="MILP solutions of week 3")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Guaranteed detection p*_F (%)", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel("Cost (units)")
    ax.set_ylabel("Weighted coverage F (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cost / coverage / worst-case front", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hv(hv: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    style(ax)
    for (g, d), col in zip(hv.groupby("seed"), C_SEEDS):
        ax.plot(d.evaluations, d.hv, color=col, lw=2, label=f"seed {g}")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume (normalised objectives)")
    ax.set_title("Convergence of NSGA-II", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_price(p: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    style(ax)
    ax.step(p.budget, 100 * p.F_max, where="post", color=C_OPT, lw=2,
            label="Best coverage, unconstrained")
    ax.step(p.budget, 100 * p["F_max_p0.95"], where="post", color=C_GREEDY, lw=2,
            label="... with p*_F ≥ 0.95")
    ax.set_xlabel("Budget (cost units)")
    ax.set_ylabel("Weighted coverage F (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Price of robustness", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Step 4: orchestration
# =============================================================================
def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    out = ROOT / cfg["nsga"]["out_dir"]
    seeds = cfg["nsga"]["seeds"]
    fronts = pd.concat([pd.read_csv(out / f"front_{g}.csv") for g in seeds], ignore_index=True)
    hv = pd.concat([pd.read_csv(out / f"hv_{g}.csv").assign(seed=g) for g in seeds])
    pareto = pd.read_csv(ROOT / cfg["optimization"]["out_dir"] / "pareto.csv")
    w3 = pd.read_csv(ROOT / cfg["intruder"]["out_dir"] / "week3.csv")
    pd.set_option("display.width", 200)

    # --- Stability across seeds
    last = hv.groupby("seed").last()
    print(f"Final hypervolume: {last.hv.mean():.4f} ± {last.hv.std():.4f} "
          f"(min {last.hv.min():.4f}, max {last.hv.max():.4f}); "
          f"mean time {last.time_s.mean():.0f} s per seed")

    # --- Validation
    front = validate(front_global(fronts), pareto)
    front.to_csv(out / "front_global.csv", index=False)
    print(f"\nGlobal front: {len(front)} solutions; none above the certified UB bound ✔")
    comp = []
    for _, r in pareto.sort_values("budget").iterrows():
        d = front[front.cost <= r.budget + 1e-9]
        comp.append(dict(budget=r.budget, lb=r.lb, ub=r.ub, nsga=d.F.max() if len(d) else 0.0))
    comp = pd.DataFrame(comp)
    comp["gap_lb_pt"] = 100 * (comp.lb - comp.nsga)
    comp["gap_ub_pt"] = 100 * (comp.ub - comp.nsga)
    comp.to_csv(out / "validation_ub.csv", index=False)
    better = comp[comp.nsga > comp.lb + 1e-6]
    print(f"Best F of the front against week 3: max gap to LB {comp.gap_lb_pt.max():.2f} pt "
          f"(B = {comp.budget[comp.gap_lb_pt.idxmax()]:g}), median {comp.gap_lb_pt.median():.2f} pt")
    print(f"Budgets where NSGA-II IMPROVES the best known solution (LB): "
          f"{[f'{b:g} (+{-e:.2f} pt)' for b, e in zip(better.budget, better.gap_lb_pt)]}")
    print(f"Residual gap to the UB bound: max {comp.gap_ub_pt.max():.2f} pt")

    # --- Week 3 dominated?
    allsol = pd.concat([front.assign(src="nsga"), w3.assign(src=w3.method)], ignore_index=True)
    nd = non_dominated(allsol)
    dom_w3 = allsol[(allsol.src != "nsga") & ~nd]
    print(f"Week 3 solutions dominated by the front: "
          f"{len(dom_w3)} / {len(w3)} (MILP: {(dom_w3.src == 'milp').sum()}, "
          f"greedy: {(dom_w3.src == 'greedy').sum()})")
    print("  dominated MILP:", sorted(dom_w3[dom_w3.src == "milp"].budget.tolist()))

    # --- Price of robustness
    sols = pd.concat([front, w3], ignore_index=True)[["cost", "F", "p_star_F"]]
    p = price(sols, np.arange(2.0, 31.0))
    p.to_csv(out / "price_of_robustness.csv", index=False)
    print("\nPrice of robustness (points of F lost to guarantee p*_F ≥ threshold):")
    print(p.round(3).to_string(index=False))

    print("\nComposition of the front (p*_F ≥ 0.95):")
    print(front[front.p_star_F >= 0.95][["cost", "F", "p_star_F", "p_star_G", "k_cov",
                                         "n_radar", "n_camera", "n_outside"]]
          .round(3).to_string(index=False))

    plot_front(front, w3, pareto, out / "front_nsga.png")
    plot_hv(hv, out / "hv.png")
    plot_price(p, out / "price_of_robustness.png")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
