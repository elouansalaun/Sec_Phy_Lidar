"""
src/nsga.py: multi-objective optimisation of the layout: NSGA-II.
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
from src.intruder import corridor_threshold, detection, load_domain, solutions_w3


# =============================================================================
# Step 1: extended instance and evaluation
# =============================================================================
def k_coverage_zones(spec, cfg, root, cell_zone) -> list[int]:
    """Indices (coverage layer) of the zones subject to k-coverage.

    spec: list of indices or names (source column), or "equipment": the approach rings of
    every equipment asset (failing that, the one of the intruder target).
    Only the zones that contain at least one cell are kept.
    """
    import geopandas as gpd

    derived = root / cfg["site"]["output_gpkg"]
    sources = list(gpd.read_file(derived, layer="coverage")["source"])
    if spec == "equipment":
        assets = gpd.read_file(derived, layer="assets")
        names = set(assets.loc[assets["type"] == "equipment", "name"])
        idx = [i for i, s in enumerate(sources) if s in names]
        if not idx:
            target = set(assets.loc[assets["intruder_target"].astype(bool), "name"])
            idx = [i for i, s in enumerate(sources) if s in target]
    else:
        idx = [int(z) if not isinstance(z, str) else sources.index(z) for z in spec]
    idx = [i for i in idx if np.any(cell_zone == i)]
    assert idx, f"k-coverage: no zone kept for {spec!r}"
    return idx


class Evaluator:
    """Everything needed to judge a set T of indices of A_intruder."""

    def __init__(self, cfg, root):
        self.cfg = cfg
        self.dom, self.A, self.conf = load_domain(cfg, root)
        cells = np.load(root / cfg["coverage"]["out_dir"] / "cells.npz")
        n_int = int(self.dom.inside.sum())
        assert self.dom.inside[:n_int].all()                  # interior first (src.intruder_domain)
        self.n_int = n_int
        self.inst = from_configs(self.A[:, :n_int], cells["w"], self.conf, cfg)
        self.zone = cells["zone"]
        k = cfg["nsga"]["k_coverage"]
        self.k, self.k_p, self.k_share = k["k"], k["p_min"], k["min_share"]
        self.k_zones = k_coverage_zones(k["zones"], cfg, root, self.zone)
        self.B_max = float(cfg["optimization"]["budget"]["max"])

    def evaluate(self, T) -> dict:
        T = np.flatnonzero(T) if np.asarray(T).dtype == bool else np.asarray(T, np.int64)
        P, _ = detection(self.A, T)
        F = float(self.inst.w @ P[:self.n_int]) / self.inst.W
        p_F = corridor_threshold(self.dom, P, "F")[0] if T.size else 0.0
        return dict(cost=cost(self.inst, T), F=F, p_star_F=p_F, k_cov=self.k_coverage(T))

    def k_coverage(self, T) -> float:
        """Smallest share, over the equipment, of the cells seen by ≥ k sensors at P ≥ p_min."""
        if len(T) == 0:
            return 0.0
        cnt = (self.A[T][:, :self.n_int].toarray() >= self.k_p).sum(axis=0)
        return float(min(np.mean(cnt[self.zone == z] >= self.k) for z in self.k_zones))


# =============================================================================
# Step 2: repair: (C3) one height per location, (C4) capacity, budget
# =============================================================================
def repair(inst, x: np.ndarray, B_max: float, rng) -> np.ndarray:
    x = x.copy()
    T = np.flatnonzero(x)
    if T.size == 0:
        return x
    # (C3): on a location with several heights, keep the one with the most sensors
    s = inst.support[T]
    site = inst.site[s]
    for c in np.unique(site):
        sup_c = np.unique(s[site == c])
        if sup_c.size > 1:
            n = np.array([(s == sp).sum() for sp in sup_c]) + rng.random(sup_c.size) * 0.5
            x[T[(site == c) & (s != sup_c[np.argmax(n)])]] = False
    # (C4): capacity per type, random removals
    T = np.flatnonzero(x)
    for sp, kd in set(zip(inst.support[T].tolist(), inst.kind[T].tolist())):
        idx = T[(inst.support[T] == sp) & (inst.kind[T] == kd)]
        if idx.size > inst.capacity[kd]:
            x[rng.choice(idx, idx.size - inst.capacity[kd], replace=False)] = False
    # Budget: random removals until cost ≤ B_max
    T = np.flatnonzero(x)
    while T.size and cost(inst, T) > B_max + 1e-9:
        x[rng.choice(T)] = False
        T = np.flatnonzero(x)
    return x


# =============================================================================
# Step 3: pymoo problem and operators
# =============================================================================
class Placement(ElementwiseProblem):
    def __init__(self, ev: Evaluator):
        self.ev = ev
        self.constrained = bool(ev.cfg["nsga"]["k_coverage"]["constraint"])
        super().__init__(n_var=ev.inst.n_configs, n_obj=3,
                         n_ieq_constr=1 if self.constrained else 0, xl=0, xu=1, vtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        r = self.ev.evaluate(x.astype(bool))
        out["F"] = [r["cost"], -r["F"], -r["p_star_F"]]
        if self.constrained:
            out["G"] = [self.ev.k_share - r["k_cov"]]


class RepairPlacement(Repair):
    def __init__(self, ev, rng):
        super().__init__()
        self.ev, self.rng = ev, rng

    def _do(self, problem, X, **kwargs):
        return np.array([repair(self.ev.inst, x.astype(bool), self.ev.B_max, self.rng)
                         for x in X])


class SeededSampling(Sampling):
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


class UnionCrossover(Crossover):
    """Child = random subset of the union of the parents (never empty)."""

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


class NeighbourhoodMutation(Mutation):
    OPS = ("add", "remove", "replace", "neighbour")

    def __init__(self, ev, rng, weights=(0.3, 0.25, 0.15, 0.3)):
        super().__init__()
        self.rng, self.p = rng, np.asarray(weights) / np.sum(weights)
        cand = ev.conf["cand"].to_numpy()
        self.neighbours = {c: np.flatnonzero(cand == c) for c in np.unique(cand)}
        self.cand = cand

    def _do(self, problem, X, **kwargs):
        X = X.astype(bool).copy()
        n = problem.n_var
        for x in X:
            for _ in range(1 + self.rng.geometric(0.5) - 1):   # 1, sometimes several operations
                T = np.flatnonzero(x)
                op = self.OPS[self.rng.choice(4, p=self.p)] if T.size else "add"
                if op == "add":
                    x[self.rng.integers(n)] = True
                elif op == "remove" and T.size > 1:
                    x[self.rng.choice(T)] = False
                elif op == "replace":
                    x[self.rng.choice(T)] = False
                    x[self.rng.integers(n)] = True
                elif op == "neighbour":
                    j = self.rng.choice(T)
                    x[j] = False
                    x[self.rng.choice(self.neighbours[self.cand[j]])] = True
        return X


class HVTracker(Callback):
    """Hypervolume of the current front, normalised objectives: (cost/B_max, −F, −p*)."""

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
# Step 4: running one seed
# =============================================================================
def run(ev: Evaluator, seed: int, root) -> tuple[pd.DataFrame, pd.DataFrame]:
    nc = ev.cfg["nsga"]
    rng = np.random.default_rng(seed)
    opt = root / ev.cfg["optimization"]["out_dir"]
    seeds = []
    if nc["seed_with_week3"]:
        for f in ("milp_solutions.csv", "greedy_solutions.csv"):
            seeds += list(solutions_w3(ev.conf, opt / f).values())
    pop = int(nc["population"])
    assert len(seeds) <= pop, "population smaller than the seeding"

    algo = NSGA2(pop_size=pop, sampling=SeededSampling(seeds, rng),
                 crossover=UnionCrossover(rng, nc["p_crossover"]),
                 mutation=NeighbourhoodMutation(ev, rng), repair=RepairPlacement(ev, rng),
                 eliminate_duplicates=True)
    cb = HVTracker(ev.B_max)
    t0 = time.time()
    res = minimize(Placement(ev), algo, ("n_gen", int(nc["generations"])), seed=seed,
                   callback=cb, verbose=False)
    dt = time.time() - t0

    rows = []
    for x in np.atleast_2d(res.X):
        T = np.flatnonzero(x.astype(bool))
        r = ev.evaluate(T)
        P, _ = detection(ev.A, T)
        r["p_star_G"] = corridor_threshold(ev.dom, P, "G")[0]
        k = ev.inst.kind[T]
        rows.append(dict(seed=seed, **r, n_radar=int((k == KINDS.index("radar")).sum()),
                         n_camera=int((k == KINDS.index("camera")).sum()),
                         n_outside=int((ev.conf["config_w3"].to_numpy()[T] < 0).sum()),
                         configs=" ".join(map(str, T))))
    front = pd.DataFrame(rows).sort_values("cost").reset_index(drop=True)
    hv = pd.DataFrame(cb.rows)
    hv["time_s"] = dt
    return front, hv


def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    nc = cfg["nsga"]
    out = ROOT / nc["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(cfg, ROOT)
    seeds = [int(a) for a in sys.argv[1:]] or nc["seeds"]
    print(f"{ev.inst.n_configs:,} configurations, B_max = {ev.B_max:g}, "
          f"k-coverage as {'constraint' if nc['k_coverage']['constraint'] else 'indicator'}")
    for g in seeds:
        front, hv = run(ev, g, ROOT)
        front.to_csv(out / f"front_{g}.csv", index=False)
        hv.to_csv(out / f"hv_{g}.csv", index=False)
        print(f"\nSeed {g}: {len(front)} non-dominated solutions, HV = {hv.hv.iloc[-1]:.4f}, "
              f"{hv.evaluations.iloc[-1]:,} evaluations, {hv.time_s.iloc[0]:.0f} s")
        print(front.drop(columns=["configs", "seed"]).round(3).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
