"""
src/milp.py — Optimum certifié par coupes sous-modulaires multi-blocs (docs/formulation.md, §4).

Le modèle compact à coupes tangentes (une variable par cellule) est correct mais intraitable ici :
3,3 M non-nuls, relaxation continue en 84 s, borne 0,927 contre 0,680 réalisable à B = 12.
On résout donc le problème exact par génération de contraintes (Nemhauser et Wolsey, 1981) :

    F = Σ_g F_g,  F_g(T) = Σ_{i∈g} w_i (1 − Π_{j∈T} (1 − P_ij))   (g : bloc spatial de cellules)
    Pour tout S, T :  F_g(T) ≤ F_g(S) + Σ_{j∈T∖S} ρ_gj(S),   ρ_gj(S) = Σ_{i∈g} w_i q_i(S) P_ij
    Maître :  max Σ_g η_g   s.c. (C1)–(C4),  η_g ≤ F_g(S_k) + Σ_{j∉S_k} ρ_gj(S_k) x_j  ∀ k, g

Le maître majore l'optimum (borne duale de HiGHS) ; chaque solution T_k, évaluée exactement,
le minore. On ajoute la coupe en T_k et on recommence jusqu'à l'écart voulu. Pas de logarithme,
pas de plafonnement de P, pas de tangentes : l'objectif est exact.

Sorties : optim/milp.csv, optim/milp_solutions.csv, optim/milp_convergence.csv
Usage   : python -m src.milp
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from src.instance import EPS, KINDS, Instance, cost, evaluate, non_detection, violations


# =============================================================================
# Étape 1 — Contraintes structurelles (C1)–(C4) sur (x, y)
# =============================================================================
def structural(inst: Instance, budget: float):
    """Matrice creuse sur les variables (x, y) et bornes des lignes."""
    nJ, nS = inst.n_configs, inst.n_supports
    J = np.arange(nJ)
    rows, lo, hi = [], [], []

    def add(Mx, My, l, u):
        M = sparse.hstack([sparse.csr_matrix(Mx), sparse.csr_matrix(My)]).tocsr()
        rows.append(M)
        lo.append(np.full(M.shape[0], l, dtype=float))
        hi.append(np.full(M.shape[0], u, dtype=float))

    # (C1) budget
    add(inst.cost_sensor[None, :], inst.cost_support[None, :], -np.inf, budget)
    # (C2) x_j − y_s(j) ≤ 0
    add(sparse.identity(nJ), sparse.coo_matrix((-np.ones(nJ), (J, inst.support)), (nJ, nS)),
        -np.inf, 0)
    # (C3) Σ_h y_(c,h) ≤ 1 (emplacements à plusieurs hauteurs)
    nC = inst.site.max() + 1
    Y = sparse.coo_matrix((np.ones(nS), (inst.site, np.arange(nS))), (nC, nS)).tocsr()
    multi = np.flatnonzero(np.diff(Y.indptr) > 1)
    if multi.size:
        add(sparse.csr_matrix((multi.size, nJ)), Y[multi], -np.inf, 1)
    # (C4) Σ_{j∈s, κ} x_j − K_κ y_s ≤ 0
    keys, r = np.unique(inst.support * len(KINDS) + inst.kind, return_inverse=True)
    s_k, k_k = keys // len(KINDS), keys % len(KINDS)
    add(sparse.coo_matrix((np.ones(nJ), (r, J)), (keys.size, nJ)),
        sparse.coo_matrix((-inst.capacity[k_k].astype(float), (np.arange(keys.size), s_k)),
                          (keys.size, nS)), -np.inf, 0)
    return sparse.vstack(rows, format="csr"), np.concatenate(lo), np.concatenate(hi)


# =============================================================================
# Étape 2 — Blocs de cellules et coupes sous-modulaires
# =============================================================================
def spatial_blocks(rows: np.ndarray, cols: np.ndarray, block_px: int) -> np.ndarray:
    """Indice de bloc carré (block_px × block_px pixels) de chaque cellule."""
    key = (rows // block_px).astype(np.int64) * (cols.max() + 1) + cols // block_px
    return np.unique(key, return_inverse=True)[1]


class Cuts:
    """Génère η_g − Σ_{j∉S} ρ_gj(S) x_j ≤ F_g(S) pour chaque bloc g."""

    def __init__(self, inst: Instance, groups: np.ndarray):
        self.inst = inst
        self.G = int(groups.max()) + 1
        nI = inst.A.shape[1]
        self.Gm = sparse.csr_matrix((np.ones(nI), (groups, np.arange(nI))), shape=(self.G, nI))
        self.AT = inst.A.T.tocsr()
        self.Wg = self.Gm @ inst.w
        self.seen: set[tuple] = set()
        self.blocks: list[sparse.csr_matrix] = []
        self.rhs: list[np.ndarray] = []

    def add(self, T) -> bool:
        """Ajoute les G coupes en S = T ; False si T est déjà dans le pool."""
        T = np.unique(np.asarray(list(T), dtype=np.int64))
        key = tuple(T)
        if key in self.seen:
            return False
        self.seen.add(key)
        inst = self.inst
        q = non_detection(inst, T)
        R = sparse.csr_matrix(self.Gm.multiply(inst.w * q) @ self.AT)     # ρ_gj(S), G × J
        keep = np.ones(inst.n_configs)
        keep[T] = 0.0
        R = R @ sparse.diags(keep)                                          # j ∈ S : ρ = 0
        self.blocks.append(sparse.hstack([-R, sparse.csr_matrix((self.G, inst.n_supports)),
                                          sparse.identity(self.G)]).tocsr())
        self.rhs.append(self.Gm @ (inst.w * (1 - q)))                        # F_g(S)
        return True

    def matrix(self):
        return sparse.vstack(self.blocks, format="csr"), np.concatenate(self.rhs)


# =============================================================================
# Étape 3 — Boucle maître / évaluation
# =============================================================================
@dataclass
class MilpResult:
    T: np.ndarray                 # meilleure solution trouvée
    value: float                  # F(T) exact : borne inférieure
    ub: float                     # borne supérieure certifiée
    cost: float
    iterations: int
    time_s: float
    converged: bool          # écart ≤ gap_tol (sinon : limite de temps ou stagnation)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def gap(self) -> float:
        return (self.ub - self.value) / self.ub if self.ub > 0 else 0.0


def solve(inst: Instance, budget: float, groups: np.ndarray, seeds=(),
          gap_tol: float = 5e-3, master_time_s: float = 120.0, max_time_s: float = 900.0,
          max_iter: int = 100, verbose: bool = False) -> MilpResult:
    nJ, nS = inst.n_configs, inst.n_supports
    n = nJ + nS
    S_A, S_lo, S_hi = structural(inst, budget)
    cuts = Cuts(inst, groups)
    G = cuts.G

    best_T, best_v = np.array([], dtype=np.int64), 0.0
    cuts.add([])
    for T in seeds:                                    # solutions connues : coupes + borne inf.
        T = np.asarray(list(T), dtype=np.int64)
        cuts.add(T)
        if not violations(inst, T, budget):
            v = evaluate(inst, T)
            if v > best_v:
                best_T, best_v = T, v

    c = np.concatenate([np.zeros(n), -np.ones(G)])
    bounds = Bounds(np.zeros(n + G), np.concatenate([np.ones(n), cuts.Wg]))
    integ = np.concatenate([np.ones(n), np.zeros(G)])
    S_block = sparse.hstack([S_A, sparse.csr_matrix((S_A.shape[0], G))]).tocsr()

    t0, ub, hist, converged = time.perf_counter(), np.inf, [], False
    for it in range(1, max_iter + 1):
        C, rhs = cuts.matrix()
        cons = LinearConstraint(sparse.vstack([S_block, C], format="csr"),
                                np.concatenate([S_lo, np.full(rhs.size, -np.inf)]),
                                np.concatenate([S_hi, rhs]))
        left = max_time_s - (time.perf_counter() - t0)
        res = milp(c, constraints=cons, bounds=bounds, integrality=integ,
                   options=dict(time_limit=max(1.0, min(master_time_s, left)), mip_rel_gap=1e-4))
        if res.x is None:
            break
        ub = min(ub, -res.mip_dual_bound)              # borne duale, valide même si time limit
        T = np.flatnonzero(res.x[:nJ] > 0.5)
        v = evaluate(inst, T)
        if v > best_v:
            best_T, best_v = T, v
        new = cuts.add(T)
        gap = (ub - best_v) / ub if ub > EPS else 0.0
        hist.append(dict(iteration=it, temps_s=time.perf_counter() - t0, ub=ub, lb=best_v,
                         F_maitre=v, ecart=gap, n_coupes=len(cuts.blocks)))
        if verbose:
            print(f"  it {it:3d} | {hist[-1]['temps_s']:6.1f} s | UB {ub / inst.W:.4f} "
                  f"| LB {best_v / inst.W:.4f} | écart {100 * gap:.2f} %", flush=True)
        if gap <= gap_tol:
            converged = True
            break
        if not new:
            # T déjà coupé : si le maître a prouvé son optimum (status 0, à mip_rel_gap près),
            # UB = F(T) et T est optimal. Sinon, le maître s'est arrêté sur sa limite de temps
            # et renverra la même solution : on arrête sans prétendre avoir convergé.
            converged = res.status == 0
            break
        if time.perf_counter() - t0 > max_time_s:
            break

    return MilpResult(T=best_T, value=best_v, ub=ub, cost=cost(inst, best_T),
                      iterations=len(hist), time_s=time.perf_counter() - t0,
                      converged=converged, history=pd.DataFrame(hist))


# =============================================================================
# Étape 4 — Balayage du budget
# =============================================================================
def prefixes(trace: pd.DataFrame):
    """Ensembles successifs construits par le glouton : autant de coupes gratuites."""
    cfgs = trace["config"].tolist() if len(trace) else []
    return [cfgs[:k] for k in range(1, len(cfgs) + 1)]


def main() -> None:
    from src.greedy import describe, greedy
    from src.instance import load_instance
    from src.site import ROOT, load_config

    cfg = load_config()
    oc, mc = cfg["optimisation"], cfg["optimisation"]["milp"]
    inst, configs = load_instance(cfg, ROOT)
    cel = np.load(ROOT / cfg["coverage"]["out_dir"] / "cellules.npz")
    block_px = int(round(mc["bloc_m"] / cfg["site"]["resolution_m"]))
    groups = spatial_blocks(cel["rows"], cel["cols"], block_px)
    out = ROOT / oc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    b = oc["budget"]
    budgets = np.arange(b["min"], b["max"] + EPS, b["pas"])
    W = inst.W
    print(f"{groups.max() + 1} blocs de {mc['bloc_m']} m, {inst.n_configs:,} configurations, "
          f"{inst.n_supports} supports\n")

    rows, sols, conv, prev = [], [], [], []
    for B in budgets:
        g = greedy(inst, B)
        seeds = prefixes(g.trace) + [g.T] + prev       # coupes valides quel que soit B
        print(f"B = {B:g}", flush=True)
        m = solve(inst, B, groups, seeds, mc["ecart_cible"], mc["temps_maitre_s"],
                  mc["temps_max_s"], verbose=True)
        prev = [m.T]
        rows.append(dict(budget=B, glouton=g.value / W, milp=m.value / W, ub=m.ub / W,
                         ecart_certifie=m.gap, ecart_glouton=(m.ub - g.value) / m.ub,
                         cout=m.cost, iterations=m.iterations, temps_s=m.time_s,
                         converge=m.converged, **describe(inst, configs, m.T)))
        sols += [dict(budget=B, config=j) for j in m.T]
        conv.append(m.history.assign(budget=B))
        r = rows[-1]
        print(f"  -> glouton {r['glouton']:.3f} | optimum ∈ [{r['milp']:.3f}, {r['ub']:.3f}] "
              f"| {r['n_radar']} R + {r['n_camera']} C, h = {r['hauteurs']}\n", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "milp.csv", index=False)
    pd.DataFrame(sols).to_csv(out / "milp_solutions.csv", index=False)
    pd.concat(conv).to_csv(out / "milp_convergence.csv", index=False)
    print(df[["budget", "glouton", "milp", "ub", "ecart_certifie", "ecart_glouton",
              "n_radar", "n_camera", "hauteurs", "iterations", "temps_s"]]
          .round(4).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
