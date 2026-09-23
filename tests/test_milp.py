"""Tests de la décomposition par coupes sous-modulaires : optimum exact sur instances jouets."""
import numpy as np
import pytest

from src.greedy import greedy, ratio_greedy
from src.instance import evaluate, non_detection, violations
from src.milp import Cuts, solve, spatial_blocks, structural
from tests.test_greedy import brute_force, make, random_instance


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("budget", [2.0, 4.0, 7.0, 12.0])
@pytest.mark.parametrize("n_groups", [1, 5, 20])
def test_optimum_exact(seed, budget, n_groups):
    """Convergence vers l'optimum de la force brute, encadré par [LB, UB], quel que soit le découpage."""
    inst = random_instance(seed)
    groups = np.arange(inst.A.shape[1]) % n_groups
    opt = brute_force(inst, budget)
    m = solve(inst, budget, groups, gap_tol=0.0)
    assert m.converged
    assert violations(inst, m.T, budget) == []
    assert m.value == pytest.approx(opt, abs=1e-7)
    assert m.value - 1e-7 <= opt <= m.ub * (1 + 1e-3) + 1e-7


@pytest.mark.parametrize("seed", range(5))
def test_coupe_valide(seed):
    """F_g(T) ≤ F_g(S) + Σ_{j∈T∖S} ρ_gj(S) pour tous S, T et chaque bloc g."""
    inst = random_instance(seed)
    rng = np.random.default_rng(seed)
    groups = np.arange(inst.A.shape[1]) % 4
    cuts = Cuts(inst, groups)
    for _ in range(20):
        S = np.flatnonzero(rng.uniform(size=8) < 0.4)
        T = np.flatnonzero(rng.uniform(size=8) < 0.4)
        cuts.blocks.clear(); cuts.rhs.clear(); cuts.seen.clear()
        cuts.add(S)
        C, rhs = cuts.matrix()
        x = np.zeros(inst.n_configs); x[T] = 1
        Fg_T = cuts.Gm @ (inst.w * (1 - non_detection(inst, T)))
        z = np.concatenate([x, np.zeros(inst.n_supports), Fg_T])   # η_g = F_g(T)
        assert np.all(C @ z <= rhs + 1e-9)


def test_bat_le_verrouillage_de_hauteur():
    """Là où le glouton au ratio verrouille le mât de 4 m, l'optimum certifié prend 10 m."""
    P = np.array([[1, 0, 0], [1, 1, 1]], dtype=float)
    inst = make(P, support=[0, 1], site=[0, 0], a=[0.1, 3.0], f=[0.1, 0.1])
    assert ratio_greedy(inst, 10.0).value == pytest.approx(1.0)
    m = solve(inst, 10.0, np.zeros(3, dtype=int), gap_tol=0.0)
    assert list(m.T) == [1] and m.value == pytest.approx(3.0)


def test_graines_ne_changent_pas_l_optimum():
    inst = random_instance(3)
    groups = np.arange(20) % 5
    seeds = [greedy(inst, b).T for b in (2.0, 7.0, 12.0)]
    a = solve(inst, 7.0, groups, gap_tol=0.0)
    b = solve(inst, 7.0, groups, seeds=seeds, gap_tol=0.0)
    assert a.value == pytest.approx(b.value)


def test_structurel_capacite_et_hauteur():
    """Les lignes (C1)–(C4) rejettent deux hauteurs au même endroit et deux radars par support."""
    inst = make(np.eye(3), support=[0, 0, 1], kind=[1, 1, 0], site=[0, 0], a=[1, 1, 1], f=[1, 1])
    M, lo, hi = structural(inst, 100.0)
    ok = lambda x, y: np.all(M @ np.concatenate([x, y]) <= hi + 1e-9)
    assert ok([1, 0, 0], [1, 0])
    assert not ok([1, 1, 0], [1, 0])        # (C4) deux radars sur le support 0
    assert not ok([1, 0, 1], [1, 1])        # (C3) deux hauteurs sur l'emplacement 0


def test_blocs_spatiaux():
    rows = np.array([0, 0, 9, 10, 25]); cols = np.array([0, 15, 9, 0, 25])
    g = spatial_blocks(rows, cols, 10)
    assert g[0] == g[2] and len(set(g)) == 4
