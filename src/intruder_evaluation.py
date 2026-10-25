"""
src/intruder_evaluation.py: the week 3 solutions against the intruder.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.instance import cost, load_instance
from src.intruder import detection, evaluate_intruder, load_domain, plot_paths, solutions_w3

C_OPT, C_GREEDY = "#2a78d6", "#eb6834"                  # same identities as src.pareto
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHODS = {"milp": ("milp_solutions.csv", "MILP (optimum of F)", C_OPT),
           "greedy": ("greedy_solutions.csv", "Greedy", C_GREEDY)}


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
# Step 1: evaluation
# =============================================================================
def evaluate_all(cfg, root):
    ic = cfg["intruder"]
    dom, A, conf = load_domain(cfg, root)
    inst, _ = load_instance(cfg, root)
    w = inst.w
    opt = root / cfg["optimization"]["out_dir"]
    to_w3 = conf["config_w3"].to_numpy()
    rows, sols = [], {}
    for m, (f, _, _) in METHODS.items():
        for B, T in solutions_w3(conf, opt / f).items():
            r = evaluate_intruder(dom, A, T, ic)
            P, _ = detection(A, T)
            rows.append(dict(method=m, budget=B, cost=cost(inst, to_w3[T]),
                             F=float(w @ P[dom.inside]) / inst.W, n_config=T.size, **r))
            sols[(m, B)] = T
    df = pd.DataFrame(rows)
    assert np.allclose(df.F[df.method == "milp"],
                       pd.read_csv(opt / "milp.csv").sort_values("budget").milp), \
        "F recomputed on A_intruder ≠ F of week 3"
    return df, sols, dom, A


# =============================================================================
# Step 2: three-criteria dominance
# =============================================================================
def non_dominated(df: pd.DataFrame, crit=("cost", "F", "p_star_F"), tol=1e-9) -> np.ndarray:
    """Cost to minimise, the other criteria to maximise. Exact duplicates stay non-dominated."""
    X = df[list(crit)].to_numpy(float).copy()
    X[:, 0] = -X[:, 0]                                  # everything as maximisation
    nd = np.ones(len(X), bool)
    for i in range(len(X)):
        ge = (X >= X[i] - tol).all(axis=1)
        gt = (X > X[i] + tol).any(axis=1)
        nd[i] = not (ge & gt).any()
    return nd


# =============================================================================
# Step 3: figures
# =============================================================================
def plot_budget(df, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)
    for ax in (a1, a2):
        style(ax)
    for m, (_, lab, col) in METHODS.items():
        d = df[df.method == m].sort_values("budget")
        a1.step(d.budget, 100 * d.F, where="post", color=col, lw=2, label=lab)
        a2.step(d.budget, 100 * d.p_star_F, where="post", color=col, lw=2,
                label=f"{lab}: determined intruder (F)")
    d = df[df.method == "milp"].sort_values("budget")
    a2.step(d.budget, 100 * d.p_star_G, where="post", color=C_OPT, lw=1.2, ls="--",
            label="MILP: intruder through the gates (G)")
    a1.set_ylabel("Weighted coverage F (%)")
    a1.set_ylim(0, 100)
    a1.set_title("Coverage grows with the budget, worst-path detection does not",
                 loc="left", fontsize=12, color=INK)
    a1.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    a2.set_ylabel("Guaranteed detection p* (%)")
    a2.set_ylim(0, 105)
    a2.set_xlabel("Budget (cost units)")
    a2.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scatter(df, path, annot={9: "B = 9", 12: "B = 12", 13: "B = 13", 16: "B = 16 and 18", 19: "B = 19"}):
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    style(ax)
    for m, (_, lab, col) in METHODS.items():
        d = df[df.method == m]
        ax.scatter(100 * d.F, 100 * d.p_star_F, s=42 if m == "milp" else 24, color=col,
                   edgecolor="white", lw=1.2, label=lab, zorder=3 if m == "greedy" else 2)
    d = df[(df.method == "milp") & df.budget.isin(annot)]
    for _, r in d.iterrows():
        ax.annotate(annot[r.budget], (100 * r.F, 100 * r.p_star_F), xytext=(-6, 7),
                    textcoords="offset points", fontsize=8.5, color=INK_2, ha="right")
    ax.set_xlabel("Weighted coverage F (%)")
    ax.set_ylabel("Guaranteed detection of the worst path p*_F (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 103)
    ax.set_title("Average coverage against worst case", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def decoupling_budget(milp: pd.DataFrame, gr: pd.DataFrame) -> float:
    """Budget shown on the corridor map: largest drop of p*_F of the coverage optimum from one budget
    to the next; failing that, largest advantage of the greedy on p*_F; failing that, the median budget."""
    common = milp.index.intersection(gr.index)
    drop = milp.p_star_F.diff().reindex(common)
    if (drop < 0).any():
        return float(drop.idxmin())
    adv = (gr.p_star_F - milp.p_star_F).reindex(common)
    if (adv > 0).any():
        return float(adv.idxmax())
    return float(common[len(common) // 2])


# =============================================================================
# Step 4: orchestration
# =============================================================================
def main() -> None:
    import geopandas as gpd

    from src.rasters import load_grid
    from src.site import ROOT, load_config

    cfg = load_config()
    out = ROOT / cfg["intruder"]["out_dir"]
    df, sols, dom, A = evaluate_all(cfg, ROOT)
    df["non_dominated"] = non_dominated(df)
    df.to_csv(out / "week3.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["budget", "cost", "F", "p_star_F", "p_star_G", "D_pi", "P_path_F_tau1",
            "P_path_F_tau5", "non_dominated"]
    for m in METHODS:
        print(f"\n=== {m} ===")
        print(df.loc[df.method == m, cols].round(3).to_string(index=False))

    milp = df[df.method == "milp"].set_index("budget")
    gr = df[df.method == "greedy"].set_index("budget")
    zero = milp[(milp.p_star_F == 0)]
    if len(zero):
        print(f"\nMILP: p*_F = 0 up to B = {zero.index.max():g} "
              f"(F then reaches {100 * zero.F.max():.0f} %)")
    drop = milp.p_star_F.diff() < -0.05
    for B in milp.index[drop]:
        print(f"MILP: p*_F drops at B = {B:g}: {milp.p_star_F.shift()[B]:.2f} -> "
              f"{milp.p_star_F[B]:.2f} for ΔF = {100 * (milp.F[B] - milp.F.shift()[B]):+.1f} pt")
    beaten = (gr.p_star_F > milp.p_star_F + 0.05) & (milp.F >= gr.F)
    print(f"Budgets where the greedy is more robust than the optimum of F: "
          f"{[f'{b:g}' for b in gr.index[beaten]]}")
    trap = milp[(milp.p_star_G - milp.p_star_F) > 0.3]
    print(f"Gates / fence gap > 0.3 (D_pi trap, §7.3): "
          f"{[f'{b:g}' for b in trap.index]}")
    print(f"Non-dominated solutions (cost, F, p*_F): {df.non_dominated.sum()} / {len(df)}; "
          f"dominated MILP: {[f'{b:g}' for b in milp.index[~milp.non_dominated]]}")

    plot_budget(df, out / "intruder_budget.png")
    plot_scatter(df, out / "F_vs_pstar.png")

    g = load_grid(ROOT / cfg["site"]["grid_yaml"])
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimeter").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="assets")
    ic = cfg["intruder"]
    B = decoupling_budget(milp, gr)
    prev = milp.index[milp.index < B]
    shown = ([("milp", prev.max())] if len(prev) else []) + [("milp", B), ("greedy", B)]
    cases = []
    for m, B in shown:
        r = df[(df.method == m) & (df.budget == B)].iloc[0]
        res, P, paths = evaluate_intruder(dom, A, sols[(m, B)], ic, paths=True)
        cases.append((f"{METHODS[m][1].split(' ')[0]}, B = {B:g}  (F = {100 * r.F:.1f} %)",
                      res, P, paths))
    plot_paths(dom, g, per, assets, cases, out / "decoupling_paths.png")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
