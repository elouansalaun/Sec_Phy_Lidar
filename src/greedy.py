"""
src/greedy.py: gain/cost ratio greedy for sensor placement (docs/formulation.md, §5).

Marginal gain of every configuration in one sparse product: Δ = A @ (w ⊙ q),
with q the current probability of non-detection. Marginal cost = sensor
+ support if it is not open yet. We return the best of the greedy solution
and the best single sensor (Khuller, Moss and Naor, 1999).

No approximation guarantee here (shared costs, height lock-in):
the MILP (src/milp.py) certifies the gap.

Outputs : optim/greedy.csv (one row per budget), optim/greedy_solutions.csv
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
    T: np.ndarray                  # indices of the selected configurations
    value: float                   # F(T), exact evaluation
    cost: float
    source: str                    # "ratio" or "singleton"
    trace: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Step 1: ratio greedy
# =============================================================================
def ratio_greedy(inst: Instance, budget: float, tol: float = 1e-12) -> Solution:
    nJ, nS = inst.n_configs, inst.n_supports
    wq = inst.w.copy()                                   # w ⊙ q, with q = 1 at the start
    chosen = np.zeros(nJ, dtype=bool)
    is_open = np.zeros(nS, dtype=bool)
    site_support = np.full(inst.site.max() + 1, -1)     # support open at each location
    count = np.zeros((nS, len(KINDS)), dtype=np.int64)
    spent, value, steps = 0.0, 0.0, []
    gain_tol = tol * inst.W

    while True:
        gain = inst.A @ wq                               # Δ_j for every j
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
        steps.append(dict(step=len(steps) + 1, config=j, support=s, opens_support=opened,
                          gain=gain[j], marginal_cost=mcost[j], ratio=ratio[j],
                          spent=spent, F=value))

    T = np.flatnonzero(chosen)
    return Solution(T=T, value=evaluate(inst, T), cost=cost(inst, T),
                    source="ratio", trace=pd.DataFrame(steps))


# =============================================================================
# Step 2: best singleton and combination
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
    """Best of the ratio greedy and the best feasible single sensor."""
    sol = ratio_greedy(inst, budget)
    single = best_singleton(inst, budget)
    if single is not None and single.value > sol.value + EPS:
        single.trace = sol.trace                         # keep the trace for diagnostics
        return single
    return sol


# =============================================================================
# Step 3: budget sweep
# =============================================================================
def describe(inst: Instance, configs: pd.DataFrame, T) -> dict:
    sel = configs.iloc[np.asarray(T)]
    kinds = pd.Series(inst.kind[np.asarray(T)]).map(dict(enumerate(KINDS)))
    S = np.unique(inst.support[np.asarray(T)])
    h = sel.drop_duplicates(["cand", "h"])["h"]
    return dict(n_radar=int((kinds == "radar").sum()), n_camera=int((kinds == "camera").sum()),
                n_supports=len(S), heights=" ".join(f"{x:g}" for x in sorted(h)))


def main() -> None:
    from src.site import ROOT, load_config

    cfg = load_config()
    inst, configs = load_instance(cfg, ROOT)
    out = ROOT / cfg["optimization"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    b = cfg["optimization"]["budget"]
    budgets = np.arange(b["min"], b["max"] + EPS, b["step"])
    print(f"Instance: {inst.n_configs:,} configurations, {inst.n_supports} supports, "
          f"{inst.A.shape[1]:,} cells, W = {inst.W:,.0f}\n")

    rows, sols = [], []
    for B in budgets:
        t0 = time.perf_counter()
        sol = greedy(inst, B)
        dt = time.perf_counter() - t0
        v = violations(inst, sol.T, B)
        assert not v, f"B = {B}: {v}"
        rows.append(dict(budget=B, F=sol.value, coverage=sol.value / inst.W, cost=sol.cost,
                         source=sol.source, n_config=sol.T.size, time_s=dt,
                         **describe(inst, configs, sol.T)))
        sols += [dict(budget=B, config=j) for j in sol.T]

    df = pd.DataFrame(rows)
    print(df.drop(columns="F").round(3).to_string(index=False))
    df.to_csv(out / "greedy.csv", index=False)
    pd.DataFrame(sols).to_csv(out / "greedy_solutions.csv", index=False)

    # Detail of the solution at the median budget: how the greedy builds its solution
    B = budgets[len(budgets) // 2]
    sol = greedy(inst, B)
    tr = sol.trace.merge(configs[["cand", "h", "sensor", "azimuth"]],
                         left_on="config", right_index=True)
    tr["coverage"] = tr["F"] / inst.W
    print(f"\nGreedy trace at B = {B:g} (selected solution: {sol.source}):")
    print(tr[["step", "cand", "h", "sensor", "azimuth", "opens_support", "marginal_cost",
              "gain", "spent", "coverage"]].round(3).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
