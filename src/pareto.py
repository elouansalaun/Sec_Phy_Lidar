"""
src/pareto.py: coverage / cost front: greedy against the certified optimum.
Reads the outputs of src.greedy and src.milp, without recomputing anything.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference palette (dataviz skill): identity per series, text in neutral ink
C_OPT, C_GREEDY = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


# =============================================================================
# Step 1: front table
# =============================================================================
def front_table(milp: pd.DataFrame, gap_tol: float) -> pd.DataFrame:
    d = milp.sort_values("budget").reset_index(drop=True)
    f = pd.DataFrame({"budget": d.budget, "greedy": d.greedy, "cost": d.cost,
                      "n_radar": d.n_radar, "n_camera": d.n_camera, "heights": d.heights})
    f["lb"] = np.maximum.accumulate(np.maximum(d.milp, d.greedy))
    f["ub"] = np.minimum.accumulate(d.ub[::-1])[::-1].to_numpy()
    f["lb_raw"], f["ub_raw"] = d.milp, d.ub
    f["certified_gap"] = (f.ub - f.lb) / f.ub
    f["greedy_loss"] = (f.lb - f.greedy) / f.lb            # proven loss (vs best known)
    f["greedy_loss_max"] = (f.ub - f.greedy) / f.ub        # possible loss (vs bound)
    f["certified"] = f.certified_gap <= gap_tol + 1e-12
    f["marginal_gain"] = f.lb.diff() / f.budget.diff()     # coverage per budget unit

    # Non-dominated solutions: no other point has a cost ≤ and a coverage ≥
    pts = d[["cost", "milp"]].to_numpy()
    f["non_dominated"] = [not np.any((pts[:, 0] <= c + 1e-9) & (pts[:, 1] > v + 1e-9))
                          for c, v in pts]
    return f


# =============================================================================
# Step 2: figures
# =============================================================================
def plot_front(f: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    style(ax)
    ax.fill_between(f.budget, 100 * f.lb, 100 * f.ub, step="post", color=C_OPT, alpha=0.18,
                    lw=0, label="Certified optimum ∈ [LB, UB]")
    ax.step(f.budget, 100 * f.lb, where="post", color=C_OPT, lw=2, label="Best solution (LB)")
    ax.step(f.budget, 100 * f.greedy, where="post", color=C_GREEDY, lw=2,
            label="Ratio greedy")
    # Radar steps: annotated on the optimal curve, at the first budget of each step
    first = f.drop_duplicates("n_radar") if f.n_radar.max() > 0 else f.iloc[:0]   # cameras only
    for _, r in first.iterrows():
        ax.annotate(f"{int(r.n_radar)} radar{'s' if r.n_radar > 1 else ''}",
                    (r.budget, 100 * r.lb), xytext=(4, 6), textcoords="offset points",
                    fontsize=8.5, color=INK_2)
    ax.set_xlabel("Budget (cost units)")
    ax.set_ylabel("Expected weighted coverage (%)")
    ax.set_title("Coverage against budget", loc="left", fontsize=12, color=INK)
    ax.set_ylim(0, 100)
    ax.set_xlim(f.budget.min(), f.budget.max())
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_gaps(f: pd.DataFrame, gap_tol: float, path):
    """Two panels, one axis each: greedy loss, gap left to certify."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), dpi=150, sharex=True)
    for ax in (a1, a2):
        style(ax)
    w = 0.7
    a1.bar(f.budget, 100 * f.greedy_loss, width=w, color=C_GREEDY,
           label="Proven loss (vs best solution)")
    a1.bar(f.budget, 100 * (f.greedy_loss_max - f.greedy_loss), bottom=100 * f.greedy_loss,
           width=w, color=C_GREEDY, alpha=0.3, label="Possible loss (up to the UB bound)")
    a1.set_ylabel("Greedy loss (%)")
    a1.set_title("What the heuristic costs, and what remains to be proven", loc="left",
                 fontsize=12, color=INK)
    a1.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)

    a2.bar(f.budget, 100 * f.certified_gap, width=w, color=C_OPT)
    a2.axhline(100 * gap_tol, color=INK_2, lw=1, ls="--")
    a2.annotate(f"target {100 * gap_tol:.1f} %", (f.budget.min(), 100 * gap_tol),
                xytext=(0, 3), textcoords="offset points", fontsize=8.5, color=INK_2)
    a2.set_ylabel("Certified gap (%)")
    a2.set_xlabel("Budget (cost units)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(conv: pd.DataFrame, budgets, path):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    style(ax)
    colors = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"]   # first 4 categorical hues
    for B, col in zip(budgets, colors):
        h = conv[conv.budget == B]
        if h.empty:
            continue
        ax.plot(h.time_s, 100 * h.gap.clip(lower=1e-4), marker="o", ms=4, lw=2, color=col,
                label=f"B = {B:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Computation time (s)")
    ax.set_ylabel("Certified gap (%, log; 0 shown at 0.01)")
    ax.set_title("Convergence of the decomposition", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Step 3: orchestration
# =============================================================================
def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    oc = cfg["optimization"]
    out = ROOT / oc["out_dir"]
    gap_tol = oc["milp"]["target_gap"]
    milp = pd.read_csv(out / "milp.csv")
    conv = pd.read_csv(out / "milp_convergence.csv")

    f = front_table(milp, gap_tol)
    f.to_csv(out / "pareto.csv", index=False)
    figs = ROOT / oc["figures_dir"]
    figs.mkdir(parents=True, exist_ok=True)

    plot_front(f, figs / "front.png")
    plot_gaps(f, gap_tol, figs / "gaps.png")
    # Budgets that really iterated: the fastest to converge, 12, 18 and the worst gap
    it = conv.groupby("budget").iteration.max()
    worst = f.nlargest(1, "certified_gap").budget.tolist()
    plot_convergence(conv, sorted({it[it >= 3].index.min(), 12.0, 18.0, *worst}),
                     figs / "convergence.png")

    pd.set_option("display.width", 200)
    cols = ["budget", "greedy", "lb", "ub", "certified_gap", "greedy_loss",
            "marginal_gain", "n_radar", "n_camera", "heights"]
    print(f[cols].round(4).to_string(index=False))
    print(f"\nCertified budgets (gap ≤ {100 * gap_tol:.1f} %): "
          f"{f.certified.sum()}/{len(f)}; max gap {100 * f.certified_gap.max():.2f} % "
          f"(B = {f.budget[f.certified_gap.idxmax()]:g})")
    print(f"Proven greedy loss: max {100 * f.greedy_loss.max():.1f} % "
          f"(B = {f.budget[f.greedy_loss.idxmax()]:g}), median "
          f"{100 * f.greedy_loss.median():.2f} %")
    tight = (f.lb_raw < f.lb - 1e-9) | (f.ub_raw > f.ub + 1e-9)
    print(f"Bounds tightened by monotonicity: {tight.sum()} budgets")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
