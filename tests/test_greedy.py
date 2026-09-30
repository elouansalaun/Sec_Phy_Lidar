"""Tests of the greedy: properties that can be checked exactly on toy instances."""
from itertools import combinations

import numpy as np
import pytest
from scipy import sparse

from src.greedy import best_singleton, greedy, ratio_greedy
from src.instance import Instance, cost, evaluate, violations

CAM, RAD = 0, 1


def make(P, w=None, support=None, kind=None, a=None, site=None, f=None, cap=(4, 1)) -> Instance:
    P = np.asarray(P, dtype=float)
    nJ, nI = P.shape
    support = np.arange(nJ) if support is None else np.asarray(support)
    nS = support.max() + 1
    return Instance(
        A=sparse.csr_matrix(P),
        w=np.ones(nI) if w is None else np.asarray(w, dtype=float),
        support=support,
        kind=np.zeros(nJ, dtype=int) if kind is None else np.asarray(kind),
        cost_sensor=np.ones(nJ) if a is None else np.asarray(a, dtype=float),
        site=np.arange(nS) if site is None else np.asarray(site),
        cost_support=np.zeros(nS) if f is None else np.asarray(f, dtype=float),
        capacity=np.asarray(cap),
    )


def random_instance(seed: int) -> Instance:
    """8 configurations on 4 supports (2 locations × 2 heights), 20 cells."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, 1, (8, 20)) * (rng.uniform(size=(8, 20)) < 0.4)
    return make(P, w=rng.uniform(0.5, 2, 20),
                support=np.repeat(np.arange(4), 2), kind=rng.integers(0, 2, 8),
                a=rng.uniform(0.5, 3, 8), site=np.array([0, 0, 1, 1]),
                f=rng.uniform(0.5, 2, 4), cap=(4, 1))


def brute_force(inst: Instance, budget: float) -> float:
    best = 0.0
    for k in range(1, inst.n_configs + 1):
        for T in combinations(range(inst.n_configs), k):
            if not violations(inst, T, budget):
                best = max(best, evaluate(inst, T))
    return best


# --- Correctness of the evaluation ---------------------------------------------------------
def test_evaluate_independence():
    """Two sensors at 0.5 on the same cell: 1 − 0.5 × 0.5 = 0.75."""
    inst = make([[0.5], [0.5]])
    assert evaluate(inst, [0, 1]) == pytest.approx(0.75)
    assert evaluate(inst, []) == 0.0


def test_log_identity():
    """1 − Π(1 − P) = 1 − exp(−Σ λ), basis of the linearisation (formulation, §4.1)."""
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 0.99, (5, 30))
    inst = make(P)
    lam = -np.log1p(-P).sum(axis=0)
    assert evaluate(inst, range(5)) == pytest.approx(np.sum(1 - np.exp(-lam)), rel=1e-12)


def test_support_cost_paid_once():
    inst = make(np.eye(2), support=[0, 0], a=[1, 1], f=[3])
    assert cost(inst, [0, 1]) == pytest.approx(5.0)


# --- Greedy -------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("budget", [2.0, 4.0, 7.0, 12.0])
def test_greedy_feasible_and_below_optimum(seed, budget):
    inst = random_instance(seed)
    sol = greedy(inst, budget)
    assert violations(inst, sol.T, budget) == []
    assert sol.value <= brute_force(inst, budget) + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_consistent_trace(seed):
    """The sum of the marginal gains Δ = A @ (w ⊙ q) gives back the exact F(T)."""
    inst = random_instance(seed)
    sol = ratio_greedy(inst, 12.0)
    assert sol.trace["gain"].sum() == pytest.approx(sol.value, rel=1e-10)


def test_optimal_on_disjoint_coverages():
    """Equal costs, disjoint coverages: the greedy is exactly optimal."""
    inst = make(np.diag([5.0, 4, 3, 2, 1]) / 5)
    sol = greedy(inst, 3.0)
    assert set(sol.T) == {0, 1, 2}
    assert sol.value == pytest.approx(brute_force(inst, 3.0))


def test_singleton_rescues_the_ratio():
    """Pathological case of the ratio alone (Khuller et al.): the singleton catches up."""
    P = np.zeros((2, 101))
    P[0, 0] = 1.0            # gain 1,   cost 0.01 -> ratio 100
    P[1, 1:] = 1.0           # gain 100, cost 10   -> ratio 10
    inst = make(P, a=[0.01, 10.0])
    assert ratio_greedy(inst, 10.0).value == pytest.approx(1.0)
    sol = greedy(inst, 10.0)
    assert sol.source == "singleton" and sol.value == pytest.approx(100.0)


def test_height_lock_in():
    """(C3): once the 4 m mast is open, the configuration at 10 m becomes forbidden."""
    P = np.array([[1, 0, 0], [1, 1, 1]], dtype=float)  # 10 m sees everything, 4 m a single cell
    inst = make(P, support=[0, 1], site=[0, 0], a=[0.1, 3.0], f=[0.1, 0.1])
    sol = ratio_greedy(inst, 10.0)
    assert list(sol.T) == [0]          # ratio 5 against 0.97: the greedy locks in 4 m
    assert brute_force(inst, 10.0) == pytest.approx(3.0)   # the optimum was 10 m


def test_radar_capacity():
    """(C4): two radars with disjoint coverages on the same support, only one allowed."""
    inst = make(np.eye(2), support=[0, 0], kind=[RAD, RAD], cap=(4, 1))
    sol = greedy(inst, 100.0)
    assert sol.T.size == 1 and violations(inst, sol.T, 100.0) == []


def test_shared_support():
    """The 2nd sensor on an already paid support only costs its price: the greedy uses it."""
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    inst = make(P, support=[0, 0, 1], a=[1, 1, 1], f=[2, 2])
    sol = greedy(inst, 4.0)
    assert set(sol.T) == {0, 1} and sol.cost == pytest.approx(4.0)


def test_insufficient_budget():
    inst = make(np.eye(2), a=[5, 5])
    assert best_singleton(inst, 1.0) is None
    assert greedy(inst, 1.0).T.size == 0
