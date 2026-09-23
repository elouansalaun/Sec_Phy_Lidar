"""
src/greedy.py — Glouton au ratio gain/coût pour le placement de capteurs (docs/formulation.md, §5).

Gain marginal de toutes les configurations en un produit creux : Δ = A @ (w ⊙ q),
avec q la probabilité de non-détection courante. Coût marginal = capteur
+ support s'il n'est pas encore ouvert. On renvoie le meilleur entre la solution
gloutonne et le meilleur capteur isolé (Khuller, Moss et Naor, 1999).

Aucune garantie d'approximation ici (coûts partagés, verrouillage de hauteur) :
le MILP (src/milp.py) certifie l'écart.

Sorties : optim/greedy.csv (une ligne par budget), optim/greedy_solutions.csv
Usage   : python -m src.greedy
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.instance import EPS, KINDS, Instance, cost, evaluate, load_instance, violations


@dataclass
class Solution:
    T: np.ndarray                  # indices des configurations retenues
    value: float                   # F(T), évaluation exacte
    cost: float
    source: str                    # "ratio" ou "singleton"
    trace: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Étape 1 — Glouton au ratio
# =============================================================================
def ratio_greedy(inst: Instance, budget: float, tol: float = 1e-12) -> Solution:
    nJ, nS = inst.n_configs, inst.n_supports
    wq = inst.w.copy()                                   # w ⊙ q, avec q = 1 au départ
    chosen = np.zeros(nJ, dtype=bool)
    is_open = np.zeros(nS, dtype=bool)
    site_support = np.full(inst.site.max() + 1, -1)     # support ouvert sur chaque emplacement
    count = np.zeros((nS, len(KINDS)), dtype=np.int64)
    spent, value, steps = 0.0, 0.0, []
    gain_tol = tol * inst.W

    while True:
        gain = inst.A @ wq                               # Δ_j pour tous les j
        mcost = inst.cost_sensor + inst.cost_support[inst.support] * ~is_open[inst.support]
        s_open = site_support[inst.site[inst.support]]
        adm = (~chosen
               & (spent + mcost <= budget + EPS)                              # (C1)
               & ((s_open == -1) | (s_open == inst.support))                  # (C3)
               & (count[inst.support, inst.kind] < inst.capacity[inst.kind])  # (C4)
               & (gain > gain_tol))
        if not adm.any():
            break
        ratio = np.where(adm, gain / mcost, -np.inf)
        j = int(np.argmax(ratio))
        s = inst.support[j]

        chosen[j] = True
        opened = not is_open[s]
        is_open[s] = True
        site_support[inst.site[s]] = s
        count[s, inst.kind[j]] += 1
        spent += mcost[j]
        value += gain[j]
        wq *= 1.0 - inst.A[j].toarray().ravel()          # q ← q ⊙ (1 − P_j·)
        steps.append(dict(etape=len(steps) + 1, config=j, support=s, ouvre_support=opened,
                          gain=gain[j], cout_marginal=mcost[j], ratio=ratio[j],
                          depense=spent, F=value))

    T = np.flatnonzero(chosen)
    return Solution(T=T, value=evaluate(inst, T), cost=cost(inst, T),
                    source="ratio", trace=pd.DataFrame(steps))


# =============================================================================
# Étape 2 — Meilleur singleton et combinaison
# =============================================================================
def best_singleton(inst: Instance, budget: float) -> Solution | None:
    single_cost = inst.cost_sensor + inst.cost_support[inst.support]
    val = np.where(single_cost <= budget + EPS, inst.A @ inst.w, -np.inf)
    if not np.isfinite(val.max()):
        return None
    j = int(np.argmax(val))
    return Solution(T=np.array([j]), value=float(val[j]), cost=float(single_cost[j]),
                    source="singleton")


def greedy(inst: Instance, budget: float) -> Solution:
    """Meilleur entre glouton au ratio et meilleur capteur isolé réalisable."""
    sol = ratio_greedy(inst, budget)
    single = best_singleton(inst, budget)
    if single is not None and single.value > sol.value + EPS:
        single.trace = sol.trace                         # garder la trace pour le diagnostic
        return single
    return sol


# =============================================================================
# Étape 3 — Balayage du budget
# =============================================================================
def describe(inst: Instance, configs: pd.DataFrame, T) -> dict:
    sel = configs.iloc[np.asarray(T)]
    kinds = pd.Series(inst.kind[np.asarray(T)]).map(dict(enumerate(KINDS)))
    S = np.unique(inst.support[np.asarray(T)])
    h = sel.drop_duplicates(["cand", "h"])["h"]
    return dict(n_radar=int((kinds == "radar").sum()), n_camera=int((kinds == "camera").sum()),
                n_supports=len(S), hauteurs=" ".join(f"{x:g}" for x in sorted(h)))


def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    inst, configs = load_instance(cfg, ROOT)
    out = ROOT / cfg["optimisation"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    b = cfg["optimisation"]["budget"]
    budgets = np.arange(b["min"], b["max"] + EPS, b["pas"])
    print(f"Instance : {inst.n_configs:,} configurations, {inst.n_supports} supports, "
          f"{inst.A.shape[1]:,} cellules, W = {inst.W:,.0f}\n")

    rows, sols = [], []
    for B in budgets:
        t0 = time.perf_counter()
        sol = greedy(inst, B)
        dt = time.perf_counter() - t0
        v = violations(inst, sol.T, B)
        assert not v, f"B = {B} : {v}"
        rows.append(dict(budget=B, F=sol.value, couverture=sol.value / inst.W, cout=sol.cost,
                         source=sol.source, n_config=sol.T.size, temps_s=dt,
                         **describe(inst, configs, sol.T)))
        sols += [dict(budget=B, config=j) for j in sol.T]

    df = pd.DataFrame(rows)
    print(df.drop(columns="F").round(3).to_string(index=False))
    df.to_csv(out / "greedy.csv", index=False)
    pd.DataFrame(sols).to_csv(out / "greedy_solutions.csv", index=False)

    # Détail de la solution au budget médian : comment le glouton construit sa solution
    B = budgets[len(budgets) // 2]
    sol = greedy(inst, B)
    tr = sol.trace.merge(configs[["cand", "h", "capteur", "azimut"]],
                         left_on="config", right_index=True)
    tr["couverture"] = tr["F"] / inst.W
    print(f"\nTrace du glouton à B = {B:g} (solution retenue : {sol.source}) :")
    print(tr[["etape", "cand", "h", "capteur", "azimut", "ouvre_support", "cout_marginal",
              "gain", "depense", "couverture"]].round(3).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
