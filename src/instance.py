"""
src/instance.py — Instance du problème de placement (docs/formulation.md, §1–3).

Regroupe ce que le glouton et le MILP partagent : les données indexées en entiers,
l'évaluation exacte de F et le contrôle des contraintes (C1)–(C4). Les deux solveurs
sont ainsi jugés par la même fonction.

Convention : A[j, i] = P_ij (configurations en lignes, cellules en colonnes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

KINDS = ("camera", "radar")
EPS = 1e-9


@dataclass(frozen=True)
class Instance:
    A: sparse.csr_matrix       # (n_configs, n_cells)
    w: np.ndarray              # (n_cells,) poids des cellules
    support: np.ndarray        # (n_configs,) indice s(j)
    kind: np.ndarray           # (n_configs,) indice dans KINDS
    cost_sensor: np.ndarray    # (n_configs,) a_j
    site: np.ndarray           # (n_supports,) emplacement c(s)
    cost_support: np.ndarray   # (n_supports,) f_s
    capacity: np.ndarray       # (len(KINDS),) K_κ

    def __post_init__(self):
        nJ, nI = self.A.shape
        assert self.w.shape == (nI,) and np.all(self.w >= 0)
        assert self.support.shape == self.kind.shape == self.cost_sensor.shape == (nJ,)
        assert self.site.shape == self.cost_support.shape
        assert np.all(self.cost_sensor > 0), "un coût nul casse le ratio gain/coût"
        assert self.A.data.size == 0 or (self.A.data.min() >= 0 and self.A.data.max() <= 1)

    @property
    def n_configs(self) -> int:
        return self.A.shape[0]

    @property
    def n_supports(self) -> int:
        return self.site.size

    @property
    def W(self) -> float:
        return float(self.w.sum())


# =============================================================================
# Construction depuis les sorties de src.coverage
# =============================================================================
def kind_map(cfg: dict) -> dict[str, str]:
    """{nom de capteur: 'camera' | 'radar'}, lu dans la section sensors (pas dans le nom)."""
    sc = cfg["sensors"]
    m = {c["name"]: "camera" for c in sc["cameras"]}
    m.update({r["name"]: "radar" for r in sc["radars"]})
    return m


def from_configs(A: sparse.spmatrix, w: np.ndarray, configs: pd.DataFrame, cfg: dict) -> Instance:
    km = kind_map(cfg)
    inconnus = set(configs["capteur"]) - set(km)
    assert not inconnus, f"capteurs absents de config.yaml : {inconnus}"

    # Support s = (emplacement, hauteur) ; son coût doit être unique
    key = configs["cand"].astype(str) + "@" + configs["h"].astype(float).map("{:g}".format)
    support, uniq = pd.factorize(key)
    sup_df = (configs.assign(_s=support)
                     .groupby("_s").agg(cand=("cand", "first"),
                                        f_min=("cout_support", "min"),
                                        f_max=("cout_support", "max")))
    assert np.allclose(sup_df.f_min, sup_df.f_max), "coût de support incohérent pour un même (c, h)"
    site, _ = pd.factorize(sup_df["cand"])

    cap = cfg["optimisation"]["capacite_support"]
    return Instance(
        A=sparse.csr_matrix(A, dtype=np.float64),
        w=np.asarray(w, dtype=np.float64),
        support=support.astype(np.int64),
        kind=configs["capteur"].map(lambda n: KINDS.index(km[n])).to_numpy(np.int64),
        cost_sensor=configs["cout_capteur"].to_numpy(np.float64),
        site=site.astype(np.int64),
        cost_support=sup_df["f_min"].to_numpy(np.float64),
        capacity=np.array([cap[k] for k in KINDS], dtype=np.int64),
    )


def load_instance(cfg: dict, root) -> tuple[Instance, pd.DataFrame]:
    d = root / cfg["coverage"]["out_dir"]
    A = sparse.load_npz(d / "A.npz")
    w = np.load(d / "cellules.npz")["w"]
    configs = pd.read_csv(d / "configs.csv")
    return from_configs(A, w, configs, cfg), configs


# =============================================================================
# Évaluation exacte et contrôle des contraintes
# =============================================================================
def as_index(T) -> np.ndarray:
    return np.unique(np.asarray(list(T), dtype=np.int64))


def non_detection(inst: Instance, T) -> np.ndarray:
    """q_i = Π_{j∈T} (1 − P_ij), calcul direct (sans plafonnement ni logarithme)."""
    T = as_index(T)
    if T.size == 0:
        return np.ones(inst.A.shape[1])
    return np.prod(1.0 - inst.A[T].toarray(), axis=0)


def evaluate(inst: Instance, T) -> float:
    """F(T) = Σ w_i (1 − q_i) : la vérité de terrain, pour tous les solveurs."""
    return float(inst.w @ (1.0 - non_detection(inst, T)))


def supports_of(inst: Instance, T) -> np.ndarray:
    return np.unique(inst.support[as_index(T)])


def cost(inst: Instance, T) -> float:
    """Σ a_j + Σ f_s, chaque support ouvert étant payé une seule fois."""
    T = as_index(T)
    return float(inst.cost_sensor[T].sum() + inst.cost_support[supports_of(inst, T)].sum())


def violations(inst: Instance, T, budget: float) -> list[str]:
    """Liste des contraintes violées ; vide si T est réalisable. (C2) tient par construction."""
    T = as_index(T)
    v = []
    c = cost(inst, T)
    if c > budget + EPS:
        v.append(f"(C1) budget : {c:.3f} > {budget:.3f}")
    S = supports_of(inst, T)
    sites, n = np.unique(inst.site[S], return_counts=True)
    for c_, k in zip(sites[n > 1], n[n > 1]):
        v.append(f"(C3) emplacement {c_} : {k} hauteurs")
    counts = np.zeros((inst.n_supports, len(KINDS)), dtype=np.int64)
    np.add.at(counts, (inst.support[T], inst.kind[T]), 1)
    for s, k in zip(*np.nonzero(counts > inst.capacity)):
        v.append(f"(C4) support {s} : {counts[s, k]} {KINDS[k]} > {inst.capacity[k]}")
    return v
