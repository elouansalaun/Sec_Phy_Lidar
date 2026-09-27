"""
src/nsga.py — Optimisation multi-objectif du dispositif : NSGA-II (étape 4.5).

Objectifs (tous minimisés par pymoo) :  coût,  −F (couverture pondérée),  −p*_F (pire cas, §7.4).
Variables : un bit par configuration de A_intrus (4 151, dont celles qui ne voient que
l'extérieur). Contraintes (C3), (C4) et budget ≤ B_max tenues par un opérateur de réparation ;
k-couverture des équipements en indicateur, ou en contrainte si nsga.k_couverture.contrainte.

Opérateurs adaptés à des solutions très creuses (≈ 2 à 10 bits sur 4 151) :
  croisement  : chaque configuration de l'union des parents est gardée avec proba 1/2
  mutation    : ajout, retrait, remplacement aléatoire, ou voisinage (même emplacement :
                autre hauteur, capteur, azimut ou inclinaison)
  réparation  : retraits aléatoires (un retrait « au plus faible gain de F » éliminerait
                d'abord les caméras extérieures, dont F est nul mais qui font p*)
Amorçage : les 58 solutions de la semaine 3 + individus aléatoires réparés.

Sorties (nsga.out_dir) : front_<graine>.csv, hv_<graine>.csv
Usage : python -m src.nsga               (toutes les graines de nsga.graines)
        python -m src.nsga 1             (une graine)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from src.instance import KINDS, cost, from_configs
from src.intrus import detection, load_domaine, seuil_couloir, solutions_s3


# =============================================================================
# Étape 1 — Instance étendue et évaluation
# =============================================================================
def zones_k_couverture(spec, cfg, root, zone_cellules) -> list[int]:
    """Indices (couche couverture) des zones soumises à la k-couverture.

    spec : liste d'indices ou de noms (colonne source), ou « equipements » : les anneaux
    d'approche de tous les actifs de type équipement (à défaut, celui de la cible de l'intrus).
    Seules les zones qui contiennent au moins une cellule sont gardées.
    """
    import geopandas as gpd

    derive = root / cfg["site"]["output_gpkg"]
    sources = list(gpd.read_file(derive, layer="couverture")["source"])
    if spec == "equipements":
        actifs = gpd.read_file(derive, layer="actifs")
        noms = set(actifs.loc[actifs["type"] == "equipement", "nom"])
        idx = [i for i, s in enumerate(sources) if s in noms]
        if not idx:
            cible = set(actifs.loc[actifs["cible_intrus"].astype(bool), "nom"])
            idx = [i for i, s in enumerate(sources) if s in cible]
    else:
        idx = [int(z) if not isinstance(z, str) else sources.index(z) for z in spec]
    idx = [i for i in idx if np.any(zone_cellules == i)]
    assert idx, f"k-couverture : aucune zone retenue pour {spec!r}"
    return idx


class Evaluateur:
    """Tout ce qu'il faut pour juger un ensemble T d'indices de A_intrus."""

    def __init__(self, cfg, root):
        self.cfg = cfg
        self.dom, self.A, self.conf = load_domaine(cfg, root)
        cells = np.load(root / cfg["coverage"]["out_dir"] / "cellules.npz")
        n_int = int(self.dom.interieur.sum())
        assert self.dom.interieur[:n_int].all()               # intérieur d'abord (src.intrusion)
        self.n_int = n_int
        self.inst = from_configs(self.A[:, :n_int], cells["w"], self.conf, cfg)
        self.zone = cells["zone"]
        k = cfg["nsga"]["k_couverture"]
        self.k, self.k_p, self.k_part = k["k"], k["p_min"], k["part_min"]
        self.k_zones = zones_k_couverture(k["zones"], cfg, root, self.zone)
        self.B_max = float(cfg["optimisation"]["budget"]["max"])

    def evaluer(self, T) -> dict:
        T = np.flatnonzero(T) if np.asarray(T).dtype == bool else np.asarray(T, np.int64)
        P, _ = detection(self.A, T)
        F = float(self.inst.w @ P[:self.n_int]) / self.inst.W
        p_F = seuil_couloir(self.dom, P, "F")[0] if T.size else 0.0
        return dict(cout=cost(self.inst, T), F=F, p_star_F=p_F, k_couv=self.k_couverture(T))

    def k_couverture(self, T) -> float:
        """Plus petite part, sur les équipements, des cellules vues par ≥ k capteurs à P ≥ p_min."""
        if len(T) == 0:
            return 0.0
        cnt = (self.A[T][:, :self.n_int].toarray() >= self.k_p).sum(axis=0)
        return float(min(np.mean(cnt[self.zone == z] >= self.k) for z in self.k_zones))


# =============================================================================
# Étape 2 — Réparation : (C3) une hauteur par emplacement, (C4) capacité, budget
# =============================================================================
def reparer(inst, x: np.ndarray, B_max: float, rng) -> np.ndarray:
    x = x.copy()
    T = np.flatnonzero(x)
    if T.size == 0:
        return x
    # (C3) : sur un emplacement portant plusieurs hauteurs, on garde celle qui a le plus de capteurs
    s = inst.support[T]
    site = inst.site[s]
    for c in np.unique(site):
        sup_c = np.unique(s[site == c])
        if sup_c.size > 1:
            n = np.array([(s == sp).sum() for sp in sup_c]) + rng.random(sup_c.size) * 0.5
            x[T[(site == c) & (s != sup_c[np.argmax(n)])]] = False
    # (C4) : capacité par type, retraits aléatoires
    T = np.flatnonzero(x)
    for sp, kd in set(zip(inst.support[T].tolist(), inst.kind[T].tolist())):
        idx = T[(inst.support[T] == sp) & (inst.kind[T] == kd)]
        if idx.size > inst.capacity[kd]:
            x[rng.choice(idx, idx.size - inst.capacity[kd], replace=False)] = False
    # Budget : retraits aléatoires jusqu'à coût ≤ B_max
    T = np.flatnonzero(x)
    while T.size and cost(inst, T) > B_max + 1e-9:
        x[rng.choice(T)] = False
        T = np.flatnonzero(x)
    return x


# =============================================================================
# Étape 3 — Problème et opérateurs pymoo
# =============================================================================
class Placement(ElementwiseProblem):
    def __init__(self, ev: Evaluateur):
        self.ev = ev
        self.contrainte = bool(ev.cfg["nsga"]["k_couverture"]["contrainte"])
        super().__init__(n_var=ev.inst.n_configs, n_obj=3,
                         n_ieq_constr=1 if self.contrainte else 0, xl=0, xu=1, vtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        r = self.ev.evaluer(x.astype(bool))
        out["F"] = [r["cout"], -r["F"], -r["p_star_F"]]
        if self.contrainte:
            out["G"] = [self.ev.k_part - r["k_couv"]]


class RepairPlacement(Repair):
    def __init__(self, ev, rng):
        super().__init__()
        self.ev, self.rng = ev, rng

    def _do(self, problem, X, **kwargs):
        return np.array([reparer(self.ev.inst, x.astype(bool), self.ev.B_max, self.rng)
                         for x in X])


class Amorcage(Sampling):
    def __init__(self, seeds: list[np.ndarray], rng, k_max: int = 8):
        super().__init__()
        self.seeds, self.rng, self.k_max = seeds, rng, k_max

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), bool)
        for i in range(n_samples):
            if i < len(self.seeds):
                X[i, self.seeds[i]] = True
            else:
                k = self.rng.integers(1, self.k_max + 1)
                X[i, self.rng.choice(problem.n_var, k, replace=False)] = True
        return X


class CroisementUnion(Crossover):
    """Enfant = sous-ensemble aléatoire de l'union des parents (jamais vide)."""

    def __init__(self, rng, prob=0.9):
        super().__init__(2, 2, prob=prob)
        self.rng = rng

    def _do(self, problem, X, **kwargs):
        _, n_mat, n_var = X.shape
        Y = np.zeros((2, n_mat, n_var), bool)
        for m in range(n_mat):
            union = np.flatnonzero(X[0, m].astype(bool) | X[1, m].astype(bool))
            for o in range(2):
                keep = union[self.rng.random(union.size) < 0.5]
                if keep.size == 0 and union.size:
                    keep = self.rng.choice(union, 1)
                Y[o, m, keep] = True
        return Y


class MutationVoisinage(Mutation):
    OPS = ("ajout", "retrait", "remplacement", "voisinage")

    def __init__(self, ev, rng, poids=(0.3, 0.25, 0.15, 0.3)):
        super().__init__()
        self.rng, self.p = rng, np.asarray(poids) / np.sum(poids)
        cand = ev.conf["cand"].to_numpy()
        self.voisins = {c: np.flatnonzero(cand == c) for c in np.unique(cand)}
        self.cand = cand

    def _do(self, problem, X, **kwargs):
        X = X.astype(bool).copy()
        n = problem.n_var
        for x in X:
            for _ in range(1 + self.rng.geometric(0.5) - 1):   # 1, parfois plusieurs opérations
                T = np.flatnonzero(x)
                op = self.OPS[self.rng.choice(4, p=self.p)] if T.size else "ajout"
                if op == "ajout":
                    x[self.rng.integers(n)] = True
                elif op == "retrait" and T.size > 1:
                    x[self.rng.choice(T)] = False
                elif op == "remplacement":
                    x[self.rng.choice(T)] = False
                    x[self.rng.integers(n)] = True
                elif op == "voisinage":
                    j = self.rng.choice(T)
                    x[j] = False
                    x[self.rng.choice(self.voisins[self.cand[j]])] = True
        return X


class SuiviHV(Callback):
    """Hypervolume du front courant, objectifs normalisés : (coût/B_max, −F, −p*)."""

    def __init__(self, B_max):
        super().__init__()
        self.B_max = B_max
        self.hv = HV(ref_point=np.array([1.05, 0.0, 0.0]))
        self.rows = []

    def notify(self, algorithm):
        Fo = algorithm.opt.get("F")
        feas = algorithm.opt.get("feasible").ravel()
        Fo = Fo[feas] if feas.any() else Fo[:0]
        val = float(self.hv(Fo * np.array([1 / self.B_max, 1, 1]))) if len(Fo) else 0.0
        self.rows.append(dict(generation=algorithm.n_gen, evaluations=algorithm.evaluator.n_eval,
                              hv=val, n_front=len(Fo)))


# =============================================================================
# Étape 4 — Lancement d'une graine
# =============================================================================
def run(ev: Evaluateur, graine: int, root) -> tuple[pd.DataFrame, pd.DataFrame]:
    nc = ev.cfg["nsga"]
    rng = np.random.default_rng(graine)
    opt = root / ev.cfg["optimisation"]["out_dir"]
    seeds = []
    if nc["amorcage_s3"]:
        for f in ("milp_solutions.csv", "greedy_solutions.csv"):
            seeds += list(solutions_s3(ev.conf, opt / f).values())
    pop = int(nc["population"])
    assert len(seeds) <= pop, "population plus petite que l'amorçage"

    algo = NSGA2(pop_size=pop, sampling=Amorcage(seeds, rng),
                 crossover=CroisementUnion(rng, nc["p_croisement"]),
                 mutation=MutationVoisinage(ev, rng), repair=RepairPlacement(ev, rng),
                 eliminate_duplicates=True)
    cb = SuiviHV(ev.B_max)
    t0 = time.time()
    res = minimize(Placement(ev), algo, ("n_gen", int(nc["generations"])), seed=graine,
                   callback=cb, verbose=False)
    dt = time.time() - t0

    rows = []
    for x in np.atleast_2d(res.X):
        T = np.flatnonzero(x.astype(bool))
        r = ev.evaluer(T)
        P, _ = detection(ev.A, T)
        r["p_star_G"] = seuil_couloir(ev.dom, P, "G")[0]
        k = ev.inst.kind[T]
        rows.append(dict(graine=graine, **r, n_radar=int((k == KINDS.index("radar")).sum()),
                         n_camera=int((k == KINDS.index("camera")).sum()),
                         n_exterieur=int((ev.conf["config_s3"].to_numpy()[T] < 0).sum()),
                         configs=" ".join(map(str, T))))
    front = pd.DataFrame(rows).sort_values("cout").reset_index(drop=True)
    hv = pd.DataFrame(cb.rows)
    hv["temps_s"] = dt
    return front, hv


def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    nc = cfg["nsga"]
    out = ROOT / nc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    ev = Evaluateur(cfg, ROOT)
    graines = [int(a) for a in sys.argv[1:]] or nc["graines"]
    print(f"{ev.inst.n_configs:,} configurations, B_max = {ev.B_max:g}, "
          f"k-couverture en {'contrainte' if nc['k_couverture']['contrainte'] else 'indicateur'}")
    for g in graines:
        front, hv = run(ev, g, ROOT)
        front.to_csv(out / f"front_{g}.csv", index=False)
        hv.to_csv(out / f"hv_{g}.csv", index=False)
        print(f"\nGraine {g} : {len(front)} solutions non dominées, HV = {hv.hv.iloc[-1]:.4f}, "
              f"{hv.evaluations.iloc[-1]:,} évaluations, {hv.temps_s.iloc[0]:.0f} s")
        print(front.drop(columns=["configs", "graine"]).round(3).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
