"""
src/evaluation_intrus.py — Les solutions de la semaine 3 face à l'intrus (étape 4.4).

Évalue toutes les solutions MILP et gloutonnes (un budget = une solution) avec src.intrus,
puis cherche les solutions dominées dès qu'on ajoute le pire cas p*_F comme troisième
critère (coût ↓, F ↑, p*_F ↑).

Sorties (intrus.out_dir) : semaine3.csv, budget_intrus.png, F_vs_pstar.png,
                           chemins_decouplage.png
Usage : python -m src.evaluation_intrus
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.instance import cost, load_instance
from src.intrus import detection, evaluer, load_domaine, plot_chemins, solutions_s3

C_OPT, C_GLOUTON = "#2a78d6", "#eb6834"                 # mêmes identités que src.pareto
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHODES = {"milp": ("milp_solutions.csv", "MILP (optimum de F)", C_OPT),
            "glouton": ("greedy_solutions.csv", "Glouton", C_GLOUTON)}


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
# Étape 1 — Évaluation
# =============================================================================
def evaluate_all(cfg, root):
    ic = cfg["intrus"]
    dom, A, conf = load_domaine(cfg, root)
    inst, _ = load_instance(cfg, root)
    w = inst.w
    opt = root / cfg["optimisation"]["out_dir"]
    to_s3 = conf["config_s3"].to_numpy()
    rows, sols = [], {}
    for m, (f, _, _) in METHODES.items():
        for B, T in solutions_s3(conf, opt / f).items():
            r = evaluer(dom, A, T, ic)
            P, _ = detection(A, T)
            rows.append(dict(methode=m, budget=B, cout=cost(inst, to_s3[T]),
                             F=float(w @ P[dom.interieur]) / inst.W, n_config=T.size, **r))
            sols[(m, B)] = T
    df = pd.DataFrame(rows)
    assert np.allclose(df.F[df.methode == "milp"],
                       pd.read_csv(opt / "milp.csv").sort_values("budget").milp), \
        "F recalculé sur A_intrus ≠ F de la semaine 3"
    return df, sols, dom, A


# =============================================================================
# Étape 2 — Dominance à trois critères
# =============================================================================
def non_dominated(df: pd.DataFrame, crit=("cout", "F", "p_star_F"), tol=1e-9) -> np.ndarray:
    """Coût à minimiser, les autres critères à maximiser. Les doublons exacts restent non dominés."""
    X = df[list(crit)].to_numpy(float).copy()
    X[:, 0] = -X[:, 0]                                  # tout en maximisation
    nd = np.ones(len(X), bool)
    for i in range(len(X)):
        ge = (X >= X[i] - tol).all(axis=1)
        gt = (X > X[i] + tol).any(axis=1)
        nd[i] = not (ge & gt).any()
    return nd


# =============================================================================
# Étape 3 — Figures
# =============================================================================
def plot_budget(df, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)
    for ax in (a1, a2):
        style(ax)
    for m, (_, lab, col) in METHODES.items():
        d = df[df.methode == m].sort_values("budget")
        a1.step(d.budget, 100 * d.F, where="post", color=col, lw=2, label=lab)
        a2.step(d.budget, 100 * d.p_star_F, where="post", color=col, lw=2,
                label=f"{lab} : intrus déterminé (F)")
    d = df[df.methode == "milp"].sort_values("budget")
    a2.step(d.budget, 100 * d.p_star_G, where="post", color=C_OPT, lw=1.2, ls="--",
            label="MILP : intrus par les portails (G)")
    a1.set_ylabel("Couverture pondérée F (%)")
    a1.set_ylim(0, 100)
    a1.set_title("La couverture croît avec le budget, la détection du pire chemin non",
                 loc="left", fontsize=12, color=INK)
    a1.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    a2.set_ylabel("Détection garantie p* (%)")
    a2.set_ylim(0, 105)
    a2.set_xlabel("Budget (unités de coût)")
    a2.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scatter(df, path, annot={9: "B = 9", 12: "B = 12", 13: "B = 13", 16: "B = 16 et 18", 19: "B = 19"}):
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    style(ax)
    for m, (_, lab, col) in METHODES.items():
        d = df[df.methode == m]
        ax.scatter(100 * d.F, 100 * d.p_star_F, s=42 if m == "milp" else 24, color=col,
                   edgecolor="white", lw=1.2, label=lab, zorder=3 if m == "glouton" else 2)
    d = df[(df.methode == "milp") & df.budget.isin(annot)]
    for _, r in d.iterrows():
        ax.annotate(annot[r.budget], (100 * r.F, 100 * r.p_star_F), xytext=(-6, 7),
                    textcoords="offset points", fontsize=8.5, color=INK_2, ha="right")
    ax.set_xlabel("Couverture pondérée F (%)")
    ax.set_ylabel("Détection garantie du pire chemin p*_F (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 103)
    ax.set_title("Couverture moyenne contre pire cas", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Étape 4 — Orchestration
# =============================================================================
def main() -> None:
    import geopandas as gpd

    from src.rasters import load_grid
    from src.site import ROOT, load_config

    cfg = load_config()
    out = ROOT / cfg["intrus"]["out_dir"]
    df, sols, dom, A = evaluate_all(cfg, ROOT)
    df["non_dominee"] = non_dominated(df)
    df.to_csv(out / "semaine3.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["budget", "cout", "F", "p_star_F", "p_star_G", "D_pi", "P_chemin_F_tau1",
            "P_chemin_F_tau5", "non_dominee"]
    for m in METHODES:
        print(f"\n=== {m} ===")
        print(df.loc[df.methode == m, cols].round(3).to_string(index=False))

    milp = df[df.methode == "milp"].set_index("budget")
    gl = df[df.methode == "glouton"].set_index("budget")
    zero = milp[(milp.p_star_F == 0)]
    print(f"\nMILP : p*_F = 0 jusqu'à B = {zero.index.max():g} "
          f"(F atteint alors {100 * zero.F.max():.0f} %)")
    chute = milp.p_star_F.diff() < -0.05
    for B in milp.index[chute]:
        print(f"MILP : p*_F chute à B = {B:g} : {milp.p_star_F.shift()[B]:.2f} -> "
              f"{milp.p_star_F[B]:.2f} pour ΔF = {100 * (milp.F[B] - milp.F.shift()[B]):+.1f} pt")
    battu = (gl.p_star_F > milp.p_star_F + 0.05) & (milp.F >= gl.F)
    print(f"Budgets où le glouton est plus robuste que l'optimum de F : "
          f"{[f'{b:g}' for b in gl.index[battu]]}")
    piege = milp[(milp.p_star_G - milp.p_star_F) > 0.3]
    print(f"Écart portails / clôture > 0,3 (piège de D_pi, §7.3) : "
          f"{[f'{b:g}' for b in piege.index]}")
    print(f"Solutions non dominées (coût, F, p*_F) : {df.non_dominee.sum()} / {len(df)} ; "
          f"MILP dominées : {[f'{b:g}' for b in milp.index[~milp.non_dominee]]}")

    plot_budget(df, out / "budget_intrus.png")
    plot_scatter(df, out / "F_vs_pstar.png")

    g = load_grid(ROOT / cfg["site"]["grid_yaml"])
    per = gpd.read_file(ROOT / cfg["site"]["gpkg"], layer="perimetre").geometry.iloc[0]
    assets = gpd.read_file(ROOT / cfg["site"]["output_gpkg"], layer="actifs")
    ic = cfg["intrus"]
    cas = []
    for m, B in (("milp", 15.0), ("milp", 16.0), ("glouton", 16.0)):
        r = df[(df.methode == m) & (df.budget == B)].iloc[0]
        res, P, paths = evaluer(dom, A, sols[(m, B)], ic, chemins=True)
        cas.append((f"{METHODES[m][1].split(' ')[0]}, B = {B:g}  (F = {100 * r.F:.1f} %)",
                    res, P, paths))
    plot_chemins(dom, g, per, assets, cas, out / "chemins_decouplage.png")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
