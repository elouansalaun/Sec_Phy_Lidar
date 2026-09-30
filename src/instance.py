"""
src/instance.py: instance of the placement problem (docs/formulation.md, §1-3).

Gathers what the greedy and the MILP share: data indexed by integers, the exact
evaluation of F and the check of constraints (C1)-(C4). Both solvers are therefore
judged by the same function.

Convention: A[j, i] = P_ij (configurations in rows, cells in columns).
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
    w: np.ndarray              # (n_cells,) cell weights
    support: np.ndarray        # (n_configs,) index s(j)
    kind: np.ndarray           # (n_configs,) index in KINDS
    cost_sensor: np.ndarray    # (n_configs,) a_j
    site: np.ndarray           # (n_supports,) location c(s)
    cost_support: np.ndarray   # (n_supports,) f_s
    capacity: np.ndarray       # (len(KINDS),) K_κ

    def __post_init__(self):
        nJ, nI = self.A.shape
        assert self.w.shape == (nI,) and np.all(self.w >= 0)
        assert self.support.shape == self.kind.shape == self.cost_sensor.shape == (nJ,)
        assert self.site.shape == self.cost_support.shape
        assert np.all(self.cost_sensor > 0), "a zero cost breaks the gain/cost ratio"
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
# Construction from the outputs of src.coverage
# =============================================================================
def kind_map(cfg: dict) -> dict[str, str]:
    """{sensor name: 'camera' | 'radar'}, read from the sensors section (not from the name)."""
    sc = cfg["sensors"]
    m = {c["name"]: "camera" for c in sc["cameras"]}
    m.update({r["name"]: "radar" for r in sc["radars"]})
    return m


def from_configs(A: sparse.spmatrix, w: np.ndarray, configs: pd.DataFrame, cfg: dict) -> Instance:
    km = kind_map(cfg)
    unknown = set(configs["sensor"]) - set(km)
    assert not unknown, f"sensors missing from config.yaml: {unknown}"

    # Support s = (location, height); its cost must be unique
    key = configs["cand"].astype(str) + "@" + configs["h"].astype(float).map("{:g}".format)
    support, uniq = pd.factorize(key)
    sup_df = (configs.assign(_s=support)
                     .groupby("_s").agg(cand=("cand", "first"),
                                        f_min=("support_cost", "min"),
                                        f_max=("support_cost", "max")))
    assert np.allclose(sup_df.f_min, sup_df.f_max), "inconsistent support cost for the same (c, h)"
    site, _ = pd.factorize(sup_df["cand"])

    cap = cfg["optimization"]["support_capacity"]
    return Instance(
        A=sparse.csr_matrix(A, dtype=np.float64),
        w=np.asarray(w, dtype=np.float64),
        support=support.astype(np.int64),
        kind=configs["sensor"].map(lambda n: KINDS.index(km[n])).to_numpy(np.int64),
        cost_sensor=configs["sensor_cost"].to_numpy(np.float64),
        site=site.astype(np.int64),
        cost_support=sup_df["f_min"].to_numpy(np.float64),
        capacity=np.array([cap[k] for k in KINDS], dtype=np.int64),
    )


def load_instance(cfg: dict, root) -> tuple[Instance, pd.DataFrame]:
    d = root / cfg["coverage"]["out_dir"]
    A = sparse.load_npz(d / "A.npz")
    w = np.load(d / "cells.npz")["w"]
    configs = pd.read_csv(d / "configs.csv")
    return from_configs(A, w, configs, cfg), configs


# =============================================================================
# Exact evaluation and constraint check
# =============================================================================
def as_index(T) -> np.ndarray:
    return np.unique(np.asarray(list(T), dtype=np.int64))


def non_detection(inst: Instance, T) -> np.ndarray:
    """q_i = Π_{j∈T} (1 − P_ij), direct computation (no capping, no logarithm)."""
    T = as_index(T)
    if T.size == 0:
        return np.ones(inst.A.shape[1])
    return np.prod(1.0 - inst.A[T].toarray(), axis=0)


def evaluate(inst: Instance, T) -> float:
    """F(T) = Σ w_i (1 − q_i): the ground truth, for every solver."""
    return float(inst.w @ (1.0 - non_detection(inst, T)))


def supports_of(inst: Instance, T) -> np.ndarray:
    return np.unique(inst.support[as_index(T)])


def cost(inst: Instance, T) -> float:
    """Σ a_j + Σ f_s, each open support being paid only once."""
    T = as_index(T)
    return float(inst.cost_sensor[T].sum() + inst.cost_support[supports_of(inst, T)].sum())


def violations(inst: Instance, T, budget: float) -> list[str]:
    """List of the violated constraints; empty if T is feasible. (C2) holds by construction."""
    T = as_index(T)
    v = []
    c = cost(inst, T)
    if c > budget + EPS:
        v.append(f"(C1) budget: {c:.3f} > {budget:.3f}")
    S = supports_of(inst, T)
    sites, n = np.unique(inst.site[S], return_counts=True)
    for c_, k in zip(sites[n > 1], n[n > 1]):
        v.append(f"(C3) location {c_}: {k} heights")
    counts = np.zeros((inst.n_supports, len(KINDS)), dtype=np.int64)
    np.add.at(counts, (inst.support[T], inst.kind[T]), 1)
    for s, k in zip(*np.nonzero(counts > inst.capacity)):
        v.append(f"(C4) support {s}: {counts[s, k]} {KINDS[k]} > {inst.capacity[k]}")
    return v
