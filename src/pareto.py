"""
src/pareto.py — Front couverture / coût : glouton contre optimum certifié (étape 3.4).

Lit les sorties de src.greedy et src.milp, sans rien recalculer.

Figures dans optimisation.figures_dir (versionné), tableau dans optimisation.out_dir.
  - front.png       : couverture en fonction du budget, glouton, meilleure solution et bande [LB, UB]
  - ecarts.png      : ce que perd le glouton, et ce qui reste à certifier
  - convergence.png : écart certifié au fil des itérations pour quelques budgets
  - pareto.csv      : front par budget, rendement marginal, solutions non dominées

Monotonie : une solution réalisable pour B l'est aussi pour tout B' ≥ B, et une borne
supérieure pour B' majore aussi l'optimum pour tout B ≤ B'. On resserre donc
LB(B) = max_{B'≤B} LB(B') et UB(B) = min_{B'≥B} UB(B').

Usage : python -m src.pareto
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette de référence (skill dataviz) : identité par série, texte en encre neutre
C_OPT, C_GLOUTON = "#2a78d6", "#eb6834"
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
# Étape 1 — Tableau du front
# =============================================================================
def front_table(milp: pd.DataFrame, gap_tol: float) -> pd.DataFrame:
    d = milp.sort_values("budget").reset_index(drop=True)
    f = pd.DataFrame({"budget": d.budget, "glouton": d.glouton, "cout": d.cout,
                      "n_radar": d.n_radar, "n_camera": d.n_camera, "hauteurs": d.hauteurs})
    f["lb"] = np.maximum.accumulate(np.maximum(d.milp, d.glouton))
    f["ub"] = np.minimum.accumulate(d.ub[::-1])[::-1].to_numpy()
    f["lb_brut"], f["ub_brut"] = d.milp, d.ub
    f["ecart_certifie"] = (f.ub - f.lb) / f.ub
    f["perte_glouton"] = (f.lb - f.glouton) / f.lb          # perte prouvée (vs meilleure connue)
    f["perte_glouton_max"] = (f.ub - f.glouton) / f.ub      # perte possible (vs borne)
    f["certifie"] = f.ecart_certifie <= gap_tol + 1e-12
    f["gain_marginal"] = f.lb.diff() / f.budget.diff()      # couverture par unité de budget

    # Solutions non dominées : aucun autre point n'a un coût ≤ et une couverture ≥
    pts = d[["cout", "milp"]].to_numpy()
    f["non_dominee"] = [not np.any((pts[:, 0] <= c + 1e-9) & (pts[:, 1] > v + 1e-9))
                        for c, v in pts]
    return f


# =============================================================================
# Étape 2 — Figures
# =============================================================================
def plot_front(f: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    style(ax)
    ax.fill_between(f.budget, 100 * f.lb, 100 * f.ub, step="post", color=C_OPT, alpha=0.18,
                    lw=0, label="Optimum certifié ∈ [LB, UB]")
    ax.step(f.budget, 100 * f.lb, where="post", color=C_OPT, lw=2, label="Meilleure solution (LB)")
    ax.step(f.budget, 100 * f.glouton, where="post", color=C_GLOUTON, lw=2,
            label="Glouton au ratio")
    # Paliers radar : annotés sur la courbe optimale, au premier budget de chaque palier
    first = f.drop_duplicates("n_radar")
    for _, r in first.iterrows():
        ax.annotate(f"{int(r.n_radar)} radar{'s' if r.n_radar > 1 else ''}",
                    (r.budget, 100 * r.lb), xytext=(4, 6), textcoords="offset points",
                    fontsize=8.5, color=INK_2)
    ax.set_xlabel("Budget (unités de coût)")
    ax.set_ylabel("Couverture pondérée espérée (%)")
    ax.set_title("Couverture en fonction du budget", loc="left", fontsize=12, color=INK)
    ax.set_ylim(0, 100)
    ax.set_xlim(f.budget.min(), f.budget.max())
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ecarts(f: pd.DataFrame, gap_tol: float, path):
    """Deux panneaux, un seul axe chacun : perte du glouton, écart restant à certifier."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), dpi=150, sharex=True)
    for ax in (a1, a2):
        style(ax)
    w = 0.7
    a1.bar(f.budget, 100 * f.perte_glouton, width=w, color=C_GLOUTON,
           label="Perte prouvée (vs meilleure solution)")
    a1.bar(f.budget, 100 * (f.perte_glouton_max - f.perte_glouton), bottom=100 * f.perte_glouton,
           width=w, color=C_GLOUTON, alpha=0.3, label="Perte possible (jusqu'à la borne UB)")
    a1.set_ylabel("Perte du glouton (%)")
    a1.set_title("Ce que coûte l'heuristique, et ce qui reste à prouver", loc="left",
                 fontsize=12, color=INK)
    a1.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)

    a2.bar(f.budget, 100 * f.ecart_certifie, width=w, color=C_OPT)
    a2.axhline(100 * gap_tol, color=INK_2, lw=1, ls="--")
    a2.annotate(f"cible {100 * gap_tol:.1f} %", (f.budget.min(), 100 * gap_tol),
                xytext=(0, 3), textcoords="offset points", fontsize=8.5, color=INK_2)
    a2.set_ylabel("Écart certifié (%)")
    a2.set_xlabel("Budget (unités de coût)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(conv: pd.DataFrame, budgets, path):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    style(ax)
    greys = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"]   # 4 premières teintes catégorielles
    for B, col in zip(budgets, greys):
        h = conv[conv.budget == B]
        if h.empty:
            continue
        ax.plot(h.temps_s, 100 * h.ecart.clip(lower=1e-4), marker="o", ms=4, lw=2, color=col,
                label=f"B = {B:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Temps de calcul (s)")
    ax.set_ylabel("Écart certifié (%, log ; 0 affiché à 0,01)")
    ax.set_title("Convergence de la décomposition", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Étape 3 — Orchestration
# =============================================================================
def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    oc = cfg["optimisation"]
    out = ROOT / oc["out_dir"]
    gap_tol = oc["milp"]["ecart_cible"]
    milp = pd.read_csv(out / "milp.csv")
    conv = pd.read_csv(out / "milp_convergence.csv")

    f = front_table(milp, gap_tol)
    f.to_csv(out / "pareto.csv", index=False)
    figs = ROOT / oc["figures_dir"]
    figs.mkdir(parents=True, exist_ok=True)

    plot_front(f, figs / "front.png")
    plot_ecarts(f, gap_tol, figs / "ecarts.png")
    # Budgets qui ont réellement itéré : le plus rapide à converger, 12, 18 et le pire écart
    it = conv.groupby("budget").iteration.max()
    worst = f.nlargest(1, "ecart_certifie").budget.tolist()
    plot_convergence(conv, sorted({it[it >= 3].index.min(), 12.0, 18.0, *worst}),
                     figs / "convergence.png")

    pd.set_option("display.width", 200)
    cols = ["budget", "glouton", "lb", "ub", "ecart_certifie", "perte_glouton",
            "gain_marginal", "n_radar", "n_camera", "hauteurs"]
    print(f[cols].round(4).to_string(index=False))
    print(f"\nBudgets certifiés (écart ≤ {100 * gap_tol:.1f} %) : "
          f"{f.certifie.sum()}/{len(f)} ; écart max {100 * f.ecart_certifie.max():.2f} % "
          f"(B = {f.budget[f.ecart_certifie.idxmax()]:g})")
    print(f"Perte prouvée du glouton : max {100 * f.perte_glouton.max():.1f} % "
          f"(B = {f.budget[f.perte_glouton.idxmax()]:g}), médiane "
          f"{100 * f.perte_glouton.median():.2f} %")
    tight = (f.lb_brut < f.lb - 1e-9) | (f.ub_brut > f.ub + 1e-9)
    print(f"Bornes resserrées par monotonie : {tight.sum()} budgets")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
