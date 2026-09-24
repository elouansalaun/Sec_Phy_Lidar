"""
src/analyse_nsga.py — Validation et lecture du front NSGA-II (étape 4.6).

  1. Validation : projeté sur (coût, F), aucun point ne doit dépasser la borne supérieure
     certifiée de la semaine 3 ; l'écart à la meilleure solution connue mesure la qualité.
  2. Stabilité : hypervolume par génération et par graine.
  3. Prix de la robustesse : meilleure couverture atteignable à budget donné, sans
     contrainte puis avec p*_F ≥ seuil.
  4. Les solutions de la semaine 3 sont-elles dominées par le front ?

Sorties (nsga.out_dir) : front_global.csv, prix_robustesse.csv, front_nsga.png,
                         hv.png, prix_robustesse.png
Usage : python -m src.analyse_nsga
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation_intrus import non_dominated, style

C_OPT, C_GLOUTON = "#2a78d6", "#eb6834"
C_SEEDS = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7", "#d4a017"]
INK, INK_2 = "#0b0b0b", "#52514e"
SEUILS = (0.9, 0.95)


# =============================================================================
# Étape 1 — Front global et validation contre la borne certifiée
# =============================================================================
def front_global(fronts: pd.DataFrame) -> pd.DataFrame:
    f = fronts.drop_duplicates("configs").reset_index(drop=True)
    return f[non_dominated(f)].sort_values("cout").reset_index(drop=True)


def borne(pareto: pd.DataFrame, c: np.ndarray, col: str) -> np.ndarray:
    """Valeur de `col` au plus petit budget de la grille ≥ c (UB croissante en B : borne valide)."""
    b = pareto.sort_values("budget")
    i = np.searchsorted(b.budget.to_numpy(), c - 1e-9)
    ok = i < len(b)
    out = np.full(c.size, np.nan)
    out[ok] = b[col].to_numpy()[i[ok]]
    return out


def validate(front: pd.DataFrame, pareto: pd.DataFrame) -> pd.DataFrame:
    front = front.copy()
    front["ub"] = borne(pareto, front.cout.to_numpy(), "ub")
    viol = front.F > front.ub + 1e-9
    assert not viol.any(), f"{viol.sum()} points au-dessus de la borne certifiée : bug"
    return front


# =============================================================================
# Étape 2 — Prix de la robustesse
# =============================================================================
def prix(sols: pd.DataFrame, budgets) -> pd.DataFrame:
    rows = []
    for B in budgets:
        d = sols[sols.cout <= B + 1e-9]
        r = dict(budget=B, F_max=d.F.max() if len(d) else 0.0)
        for s in SEUILS:
            dd = d[d.p_star_F >= s]
            r[f"F_max_p{s:g}"] = dd.F.max() if len(dd) else np.nan
        # la solution de meilleur F est-elle robuste ?
        r["p_star_du_F_max"] = d.loc[d.F.idxmax(), "p_star_F"] if len(d) else 0.0
        rows.append(r)
    p = pd.DataFrame(rows)
    for s in SEUILS:
        p[f"prix_p{s:g}_pt"] = 100 * (p.F_max - p[f"F_max_p{s:g}"])
    return p


# =============================================================================
# Étape 3 — Figures
# =============================================================================
def plot_front(front, s3, pareto, path):
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    style(ax)
    b = pareto.sort_values("budget")
    ax.fill_between(b.budget, 100 * b.lb, 100 * b.ub, step="post", color=C_OPT, alpha=0.15, lw=0,
                    label="Semaine 3 : optimum certifié de F ∈ [LB, UB]")
    sc = ax.scatter(front.cout, 100 * front.F, c=100 * front.p_star_F, cmap="Blues", vmin=0,
                    vmax=100, s=38, edgecolor=INK_2, lw=0.5, zorder=3, label="Front NSGA-II")
    m = s3[s3.methode == "milp"]
    ax.scatter(m.cout, 100 * m.F, marker="x", s=26, color=C_GLOUTON, lw=1.4, zorder=4,
               label="Solutions MILP de la semaine 3")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Détection garantie p*_F (%)", color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel("Coût (unités)")
    ax.set_ylabel("Couverture pondérée F (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Front coût / couverture / pire cas", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hv(hv: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    style(ax)
    for (g, d), col in zip(hv.groupby("graine"), C_SEEDS):
        ax.plot(d.evaluations, d.hv, color=col, lw=2, label=f"graine {g}")
    ax.set_xlabel("Évaluations")
    ax.set_ylabel("Hypervolume (objectifs normalisés)")
    ax.set_title("Convergence de NSGA-II", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_prix(p: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    style(ax)
    ax.step(p.budget, 100 * p.F_max, where="post", color=C_OPT, lw=2,
            label="Meilleure couverture, sans contrainte")
    ax.step(p.budget, 100 * p["F_max_p0.95"], where="post", color=C_GLOUTON, lw=2,
            label="… avec p*_F ≥ 0,95")
    ax.set_xlabel("Budget (unités de coût)")
    ax.set_ylabel("Couverture pondérée F (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Prix de la robustesse", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Étape 4 — Orchestration
# =============================================================================
def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    out = ROOT / cfg["nsga"]["out_dir"]
    graines = cfg["nsga"]["graines"]
    fronts = pd.concat([pd.read_csv(out / f"front_{g}.csv") for g in graines], ignore_index=True)
    hv = pd.concat([pd.read_csv(out / f"hv_{g}.csv").assign(graine=g) for g in graines])
    pareto = pd.read_csv(ROOT / cfg["optimisation"]["out_dir"] / "pareto.csv")
    s3 = pd.read_csv(ROOT / cfg["intrus"]["out_dir"] / "semaine3.csv")
    pd.set_option("display.width", 200)

    # --- Stabilité entre graines
    fin = hv.groupby("graine").last()
    print(f"Hypervolume final : {fin.hv.mean():.4f} ± {fin.hv.std():.4f} "
          f"(min {fin.hv.min():.4f}, max {fin.hv.max():.4f}) ; "
          f"temps moyen {fin.temps_s.mean():.0f} s par graine")

    # --- Validation
    front = validate(front_global(fronts), pareto)
    front.to_csv(out / "front_global.csv", index=False)
    print(f"\nFront global : {len(front)} solutions ; aucune au-dessus de la borne UB certifiée ✔")
    comp = []
    for _, r in pareto.sort_values("budget").iterrows():
        d = front[front.cout <= r.budget + 1e-9]
        comp.append(dict(budget=r.budget, lb=r.lb, ub=r.ub, nsga=d.F.max() if len(d) else 0.0))
    comp = pd.DataFrame(comp)
    comp["ecart_lb_pt"] = 100 * (comp.lb - comp.nsga)
    comp["ecart_ub_pt"] = 100 * (comp.ub - comp.nsga)
    comp.to_csv(out / "validation_ub.csv", index=False)
    mieux = comp[comp.nsga > comp.lb + 1e-6]
    print(f"Meilleur F du front face à la semaine 3 : écart à LB max {comp.ecart_lb_pt.max():.2f} pt "
          f"(B = {comp.budget[comp.ecart_lb_pt.idxmax()]:g}), médiane {comp.ecart_lb_pt.median():.2f} pt")
    print(f"Budgets où NSGA-II AMÉLIORE la meilleure solution connue (LB) : "
          f"{[f'{b:g} (+{-e:.2f} pt)' for b, e in zip(mieux.budget, mieux.ecart_lb_pt)]}")
    print(f"Écart résiduel à la borne UB : max {comp.ecart_ub_pt.max():.2f} pt")

    # --- Semaine 3 dominée ?
    allsol = pd.concat([front.assign(src="nsga"), s3.assign(src=s3.methode)], ignore_index=True)
    nd = non_dominated(allsol)
    dom_s3 = allsol[(allsol.src != "nsga") & ~nd]
    print(f"Solutions de la semaine 3 dominées par le front : "
          f"{len(dom_s3)} / {len(s3)} (MILP : {(dom_s3.src == 'milp').sum()}, "
          f"glouton : {(dom_s3.src == 'glouton').sum()})")
    print("  MILP dominées :", sorted(dom_s3[dom_s3.src == "milp"].budget.tolist()))

    # --- Prix de la robustesse
    sols = pd.concat([front, s3], ignore_index=True)[["cout", "F", "p_star_F"]]
    p = prix(sols, np.arange(2.0, 31.0))
    p.to_csv(out / "prix_robustesse.csv", index=False)
    print("\nPrix de la robustesse (points de F perdus pour garantir p*_F ≥ seuil) :")
    print(p.round(3).to_string(index=False))

    print("\nComposition du front (p*_F ≥ 0,95) :")
    print(front[front.p_star_F >= 0.95][["cout", "F", "p_star_F", "p_star_G", "k_couv",
                                         "n_radar", "n_camera", "n_exterieur"]]
          .round(3).to_string(index=False))

    plot_front(front, s3, pareto, out / "front_nsga.png")
    plot_hv(hv, out / "hv.png")
    plot_prix(p, out / "prix_robustesse.png")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
